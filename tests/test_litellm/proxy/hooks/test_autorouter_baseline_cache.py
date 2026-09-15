import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # inject the native provider's existing HTTP client owner
)
from litellm.proxy.hooks.autorouter_baseline_cache import (
    AutoRouterBaselineCache,
    BaselineCacheContext,
    cancel_baseline_cache,
    finalize_baseline_cache,
)
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimator,
    BaselineReservation,
    unknown_estimate,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router import Router
from litellm.types.router import RetryPolicy
from litellm.types.utils import CallTypes, ModelResponse, StandardLoggingRoutingDecision

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OBJECTS: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_EVENTS_ADAPTER: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])

_EVENTS: Final = _EVENTS_ADAPTER.validate_json("""
[
  {
    "type": "message_start",
    "message": {
      "id": "msg_baseline_test",
      "type": "message",
      "role": "assistant",
      "model": "claude-sonnet-5",
      "content": [],
      "stop_reason": null,
      "stop_sequence": null,
      "usage": {
        "input_tokens": 1000,
        "output_tokens": 0,
        "cache_creation_input_tokens": 5000,
        "cache_read_input_tokens": 0,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 0,
          "ephemeral_1h_input_tokens": 5000
        }
      }
    }
  },
  {
    "type": "content_block_start",
    "index": 0,
    "content_block": {
      "type": "text",
      "text": ""
    }
  },
  {
    "type": "content_block_delta",
    "index": 0,
    "delta": {
      "type": "text_delta",
      "text": "OK"
    }
  },
  {
    "type": "content_block_stop",
    "index": 0
  },
  {
    "type": "message_delta",
    "delta": {
      "stop_reason": "end_turn",
      "stop_sequence": null
    },
    "usage": {
      "output_tokens": 10
    }
  },
  {
    "type": "message_stop"
  }
]
""")

_COMPLETED: Final = _JSON_OBJECT.validate_json("""
{
  "id": "msg_baseline_test",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-5",
  "content": [
    {
      "type": "text",
      "text": "OK"
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1000,
    "output_tokens": 10,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 0,
    "cache_creation": {
      "ephemeral_5m_input_tokens": 0,
      "ephemeral_1h_input_tokens": 5000
    }
  }
}
""")

_MODELS_JSON: Final = """
[
  {
    "model_name": "opus",
    "litellm_params": {
      "model": "anthropic/claude-opus-5",
      "api_key": "test-baseline"
    },
    "model_info": {
      "id": "baseline"
    }
  }
]
"""

_MESSAGES_JSON: Final = """
[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "stable",
        "cache_control": {
          "type": "ephemeral",
          "ttl": "1h"
        }
      },
      {
        "type": "text",
        "text": "question"
      }
    ]
  }
]
"""


@runtime_checkable
class _NativeStream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...


async def _consume_stream(stream: _NativeStream) -> tuple[bytes, ...]:
    return tuple([chunk async for chunk in stream])


class _Capture(CustomLogger):
    def __init__(self, call_ids: frozenset[str]) -> None:
        self.payloads: asyncio.Queue[Mapping[str, object]] = asyncio.Queue()
        self.call_ids = call_ids

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("litellm_call_id") not in self.call_ids:
            return
        payload: Final = _OBJECTS.validate_python(kwargs.get("standard_logging_object"))
        self.payloads.put_nowait(payload)


async def _count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
    assert model == "claude-opus-5"
    return 6000 if "question" in json.dumps(_JSON_OBJECT.validate_python(body)) else 5000


def _upstream(request: httpx.Request) -> httpx.Response:
    if _JSON_OBJECT.validate_json(request.content).get("stream") is True:
        return httpx.Response(
            200,
            content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in _EVENTS),
            headers=MappingProxyType({"content-type": "text/event-stream"}),
            request=request,
        )
    return httpx.Response(200, json=_COMPLETED, request=request)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("trusted_stamp", (True, False))
async def test_native_dispatch_reserves_before_upstream_and_stamps_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    trusted_stamp: bool,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    router: Final = Router(model_list=_MESSAGES.validate_json(_MODELS_JSON))

    def get_router() -> Router:
        return router

    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    request_prefix: Final = f"native-baseline-{stream}-{trusted_stamp}"
    capture: Final = _Capture(frozenset((f"{request_prefix}-0", f"{request_prefix}-1")))
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: LiteLLM mutates its callback registries
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: LiteLLM mutates its callback registries
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_upstream)) as transport:
        client.client = transport

        async def turn(index: int) -> None:
            request_kwargs: Final = _OBJECTS.validate_json(
                '{"litellm_metadata":{"user_api_key_hash":"test-caller-hash"},"litellm_session_id":"baseline-session"}'
            )
            Router._record_routing_decision(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # exercise the production router stamp owner with its legacy kwargs contract
                request_kwargs,
                StandardLoggingRoutingDecision(
                    router_model_name="test-router",
                    router_type="complexity",
                    routed_model="sonnet",
                    cause="heuristic_scorer",
                    conversation_continuing=True,
                    savings_baseline_model="anthropic/claude-opus-5",
                    savings_baseline_deployment_id="baseline",
                ),
            )
            metadata: Final = _OBJECTS.validate_python(request_kwargs["litellm_metadata"])
            sent_metadata: Final = (
                _OBJECTS.validate_python(
                    MappingProxyType(
                        {
                            **metadata,
                            "_autorouter_baseline_route": _OBJECTS.validate_json(
                                '{"router_name":"test-router","baseline_model":"anthropic/claude-opus-5","baseline_deployment_id":"baseline"}'
                            ),
                        }
                    )
                )
                if not trusted_stamp
                else metadata
            )
            response: Final[object] = await litellm.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # the native SDK entrypoint has legacy untyped kwargs
                model="anthropic/claude-sonnet-5",
                api_key="test-selected",
                max_tokens=16,
                stream=stream,
                messages=_MESSAGES.validate_json(_MESSAGES_JSON),
                client=client,
                litellm_metadata=sent_metadata,
                litellm_session_id="baseline-session",
                litellm_call_id=f"{request_prefix}-{index}",
            )
            if stream:
                assert isinstance(response, _NativeStream)
                chunks: Final = await _consume_stream(response)
                assert chunks
            payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=20)
            estimate: Final = _OBJECTS.validate_python(payload.get("autorouter_savings_estimate"))
            if not trusted_stamp:
                assert estimate["status"] == "unknown", estimate
                assert payload["autorouter_savings"] is None
            elif index == 0:
                assert estimate["status"] == "unknown", estimate
                assert estimate["reason"] == "history_unavailable", estimate
                assert payload["autorouter_savings"] is None
            else:
                assert estimate["status"] == "estimated", estimate
                assert estimate["cache_read_input_tokens"] == 5000
                assert estimate["cache_creation_1h_input_tokens"] == 0
                saving: Final = payload["autorouter_savings"]
                assert isinstance(saving, float) and saving < 0

        await turn(0)
        await turn(1)


_RETRY_MODELS_JSON: Final = """
[
  {
    "model_name": "test-router",
    "litellm_params": {
      "model": "auto_router/complexity_router",
      "complexity_router_config": {
        "tiers": {"SIMPLE": "sonnet", "MEDIUM": "sonnet", "COMPLEX": "sonnet", "REASONING": "opus"},
        "session_affinity": false
      }
    }
  },
  {
    "model_name": "sonnet",
    "litellm_params": {"model": "anthropic/claude-sonnet-5", "api_key": "test-selected"},
    "model_info": {"id": "selected"}
  },
  {
    "model_name": "opus",
    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "test-baseline"},
    "model_info": {"id": "baseline"}
  }
]
"""


class _AttemptCapture(_Capture):
    def __init__(self, call_ids: frozenset[str]) -> None:
        super().__init__(call_ids)
        self.attempts: tuple[Logging, ...] = ()

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if (
            isinstance(logging_obj, Logging)
            and logging_obj.litellm_call_id in self.call_ids
            and call_type == CallTypes.anthropic_messages
        ):
            self.attempts = (*self.attempts, logging_obj)  # rebind-ok: record the native dispatch attempts


def _logging(request_id: str, stream: bool = False) -> Logging:
    return Logging(  # pyright: ignore[reportUnknownMemberType]  # legacy constructor owns request logging state
        model="anthropic/claude-sonnet-5",
        messages=_MESSAGES.validate_json(_MESSAGES_JSON),
        stream=stream,
        call_type=CallTypes.anthropic_messages.value,
        start_time=datetime.now(),  # noqa: DTZ005  # native Logging uses naive timestamps throughout request timing
        litellm_call_id=request_id,
        function_id=request_id,
        kwargs=_OBJECTS.validate_json('{"litellm_session_id":"baseline-session"}'),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_native_router_retry_with_shared_logging_is_unknown_and_cannot_warm_baseline(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    router: Final = Router(
        model_list=_MESSAGES.validate_json(_RETRY_MODELS_JSON),
        num_retries=1,
        retry_policy=RetryPolicy(RateLimitErrorRetries=1),
        disable_cooldowns=True,
    )

    def get_router() -> Router:
        return router

    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    request_id: Final = f"native-router-retry-{stream}"
    following_id: Final = f"native-after-retry-{stream}"
    capture: Final = _AttemptCapture(frozenset((request_id, following_id)))
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    requests: Final[list[httpx.Request]] = []  # mutable-ok: transport records the wire attempts in order

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                text='{"type":"error","error":{"type":"rate_limit_error","message":"retry"}}',
                headers=MappingProxyType({"retry-after": "0"}),
                request=request,
            )
        return _upstream(request)

    shared_logging: Final = _logging(request_id, stream)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport

        async def call(logging_obj: Logging) -> Mapping[str, object]:
            response: Final[object] = await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # dynamic router endpoint exercises the real retry loop
                model="test-router",
                max_tokens=16,
                stream=stream,
                messages=_MESSAGES.validate_json(_MESSAGES_JSON),
                litellm_logging_obj=logging_obj,
                litellm_metadata=_OBJECTS.validate_json('{"user_api_key_hash":"test-caller-hash"}'),
                litellm_session_id="baseline-session",
            )
            if stream:
                assert isinstance(response, _NativeStream)
                assert await _consume_stream(response)
            return await asyncio.wait_for(capture.payloads.get(), timeout=20)

        retried: Final = await call(shared_logging)
        assert len(requests) == 2
        assert capture.attempts == (shared_logging, shared_logging)
        assert retried["autorouter_savings"] is None
        assert _OBJECTS.validate_python(retried["autorouter_savings_estimate"])["reason"] == "retried_request"
        assert shared_logging.baseline_cache_context is None
        following: Final = await call(_logging(following_id, stream))
        assert len(requests) == 3
        assert following["autorouter_savings"] is None
        assert _OBJECTS.validate_python(following["autorouter_savings_estimate"])["reason"] == "history_unavailable"


@pytest.mark.asyncio
async def test_native_thinking_repair_retry_invalidates_baseline_without_changing_logging_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    router: Final = Router(model_list=_MESSAGES.validate_json(_RETRY_MODELS_JSON), num_retries=0)

    def get_router() -> Router:
        return router

    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    capture: Final = _AttemptCapture(frozenset(("native-thinking-repair",)))
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    requests: Final[list[httpx.Request]] = []  # mutable-ok: compare the original and repaired wire bodies

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                400,
                text='{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: Invalid `signature` in `thinking` block"}}',
                request=request,
            )
        return _upstream(request)

    shared_logging: Final = _logging("native-thinking-repair")
    messages: Final = _MESSAGES.validate_json("""
[
  {"role":"user","content":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
  {"role":"assistant","content":[{"type":"thinking","thinking":"reasoning","signature":"invalid"},{"type":"text","text":"answer"}]},
  {"role":"user","content":"question"}
]
""")
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport
        await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # native HTTP owner performs the repair retry
            model="test-router",
            max_tokens=16,
            messages=messages,
            litellm_logging_obj=shared_logging,
            litellm_metadata=_OBJECTS.validate_json('{"user_api_key_hash":"test-caller-hash"}'),
            litellm_session_id="baseline-session",
        )
        payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=20)
    assert len(requests) == 2
    assert b'"signature": "invalid"' in requests[0].content
    assert b'"signature"' not in requests[1].content
    assert capture.attempts == (shared_logging,)
    assert payload["autorouter_savings"] is None
    assert _OBJECTS.validate_python(payload["autorouter_savings_estimate"])["reason"] == "retried_request"
    assert shared_logging.baseline_cache_context is None


async def _reservation(estimator: BaselineCacheEstimator, request_id: str) -> BaselineReservation:
    reservation: Final = await estimator.reserve(
        caller_key_hash="test-caller-hash",
        session_id="baseline-session",
        router_id="test-router",
        baseline_deployment_id="baseline",
        target=NativePredictionTarget("claude-opus-5", "test-baseline"),
        request_id=request_id,
    )
    assert isinstance(reservation, BaselineReservation)
    return reservation


class _Clock:
    def __init__(self) -> None:
        self.now: float = 1000.0
        self.failures_remaining = 0

    def __call__(self) -> float:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ValueError("injected estimator clock failure")
        return self.now


async def _native_logging(
    estimator: BaselineCacheEstimator, clock: _Clock, request_id: str
) -> Logging:
    logging_obj: Final = _logging(request_id)
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, await _reservation(estimator, request_id))
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # native Logging uses naive request timestamps
    logging_obj.completion_start_time = timestamp
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType({"model": "claude-sonnet-5", "messages": _MESSAGES.validate_json(_MESSAGES_JSON)})
        ),
    )
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # native response evidence for real Logging
        httpx_response=_upstream(wire),
        api_call_start_time=timestamp,
        completion_start_time=timestamp,
        custom_llm_provider="anthropic",
        response_cost=0.125,
        stream=False,
    )
    return logging_obj


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_finalize_failure_invalidates_pending_reservation_and_preserves_spend_logging(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    logging_obj: Final = await _native_logging(estimator, clock, "finalizer-fault")
    capture: Final = _Capture(frozenset(("finalizer-fault",)))
    monkeypatch.setattr(litellm, "callbacks", (capture,))
    monkeypatch.setattr(litellm, "success_callback", ())
    monkeypatch.setattr(litellm, "_async_success_callback", (capture,))
    clock.failures_remaining = 2 if cleanup_fails else 1
    await logging_obj.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=5)
    assert payload["response_cost"] == 0.125
    assert logging_obj.baseline_cache_estimate == unknown_estimate("estimator_unavailable")
    if cleanup_fails:
        retained: Final = logging_obj.baseline_cache_context
        assert retained is not None and retained.invalidated
        await logging_obj._prepare_baseline_cache_estimate(ModelResponse())  # pyright: ignore[reportPrivateUsage]  # retry the real Logging cleanup boundary
    assert logging_obj.baseline_cache_context is None
    clock.now = 1001.0
    following: Final = await _native_logging(estimator, clock, "after-finalizer-fault")
    await following.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    assert following.baseline_cache_estimate == unknown_estimate("history_unavailable")
    clock.now = 1002.0
    matching: Final = await _native_logging(estimator, clock, "matching-after-finalizer-fault")
    await matching.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    estimate: Final = matching.baseline_cache_estimate
    assert estimate is not None and estimate.status == "estimated"
    assert estimate.cache_read_input_tokens == 5000


@pytest.mark.asyncio
async def test_late_finalize_failure_cleans_original_reservation_without_overwriting_replacement() -> None:
    clock: Final = _Clock()
    counting: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def delayed_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counting.set()
        await release.wait()
        return await _count(model, api_key, body)

    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=delayed_count, clock=clock)
    logging_obj: Final = await _native_logging(estimator, clock, "late-finalizer-fault")
    replacement: Final = BaselineCacheContext(estimator, await _reservation(estimator, "replacement-after-fault"))
    sentinel: Final = unknown_estimate("replacement_estimate")
    finalizing: Final = asyncio.create_task(
        logging_obj._prepare_baseline_cache_estimate(ModelResponse())  # pyright: ignore[reportPrivateUsage]  # inject a delayed error through the real Logging owner
    )
    await asyncio.wait_for(counting.wait(), timeout=5)
    logging_obj.baseline_cache_context = replacement  # rebind-ok: new context takes ownership while the old finalizer waits
    logging_obj.baseline_cache_estimate = sentinel
    clock.failures_remaining = 1
    release.set()
    await asyncio.wait_for(finalizing, timeout=5)
    assert logging_obj.baseline_cache_context is replacement
    assert logging_obj.baseline_cache_estimate is sentinel
    clock.now = 1001.0
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType({"model": "claude-sonnet-5", "messages": _MESSAGES.validate_json(_MESSAGES_JSON)})
        ),
    )
    recovered: Final = await estimator.finalize(
        replacement.reservation, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    assert recovered == unknown_estimate("history_unavailable")


@pytest.mark.asyncio
async def test_late_finalizer_cannot_overwrite_retry_invalidation() -> None:
    counting: Final = asyncio.Event()
    release_count: Final = asyncio.Event()
    clock: Final = _Clock()

    async def delayed_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counting.set()
        await release_count.wait()
        return await _count(model, api_key, body)

    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=delayed_count, clock=clock)
    reservation: Final = await _reservation(estimator, "late-finalizer")
    logging_obj: Final = _logging("late-finalizer")
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, reservation)
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # match native request timing evidence
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType({"model": "claude-sonnet-5", "messages": _MESSAGES.validate_json(_MESSAGES_JSON)})
        ),
    )
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # populate the real native response evidence
        httpx_response=_upstream(wire),
        api_call_start_time=timestamp,
        completion_start_time=timestamp,
        custom_llm_provider="anthropic",
        stream=False,
    )
    finalizing: Final = asyncio.create_task(finalize_baseline_cache(logging_obj, ModelResponse()))
    counting_started: Final = asyncio.create_task(counting.wait())
    await asyncio.wait((finalizing, counting_started), timeout=5, return_when=asyncio.FIRST_COMPLETED)
    if finalizing.done():
        counting_started.cancel()
        await finalizing
    assert counting.is_set(), logging_obj.baseline_cache_estimate
    await logging_obj.invalidate_baseline_cache_estimate("retried_request")
    invalidated: Final = logging_obj.baseline_cache_context
    assert invalidated is not None and invalidated.invalidated
    release_count.set()
    await asyncio.wait_for(finalizing, timeout=5)
    assert logging_obj.baseline_cache_context is invalidated
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    clock.now = 1200.0
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    clock.now = 4601.0
    following: Final = await _reservation(estimator, "after-retry-start-ttl")
    estimate: Final = await estimator.finalize(
        following, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    assert estimate.status == "unknown"
    assert estimate.reason == "history_unavailable"


class _DelayedCancelEstimator(BaselineCacheEstimator):
    def __init__(self) -> None:
        super().__init__(DualCache(), token_counter=_count)
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(self, reservation: BaselineReservation) -> None:
        self.cancel_started.set()
        await self.release_cancel.wait()
        await super().cancel(reservation)


@pytest.mark.asyncio
async def test_late_cancel_cannot_clear_replacement_context() -> None:
    estimator: Final = _DelayedCancelEstimator()
    original: Final = BaselineCacheContext(estimator, await _reservation(estimator, "original"))
    replacement: Final = BaselineCacheContext(estimator, await _reservation(estimator, "replacement"))
    logging_obj: Final = _logging("late-cancel")
    logging_obj.baseline_cache_context = original
    cancelling: Final = asyncio.create_task(cancel_baseline_cache(logging_obj))
    await asyncio.wait_for(estimator.cancel_started.wait(), timeout=5)
    logging_obj.baseline_cache_context = replacement  # rebind-ok: simulate newer work while old cancellation waits
    estimator.release_cancel.set()
    assert await asyncio.wait_for(cancelling, timeout=5) is False
    assert logging_obj.baseline_cache_context is replacement
    assert logging_obj.baseline_cache_estimate is None


@pytest.mark.asyncio
async def test_partial_stream_callback_keeps_retry_scope_until_completed_stream() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    reservation: Final = await _reservation(estimator, "partial-native-stream")
    logging_obj: Final = _logging("partial-native-stream", stream=True)
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, reservation)
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType(
                {"model": "claude-sonnet-5", "stream": True, "messages": _MESSAGES.validate_json(_MESSAGES_JSON)}
            )
        ),
    )
    initial: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # match native request timing evidence
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # simulate the native partial-stream callback evidence
        httpx_response=_upstream(wire),
        api_call_start_time=initial,
        completion_start_time=initial,
        custom_llm_provider="anthropic",
        stream=True,
        prompt_cache_response_complete=False,
    )
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_estimate == unknown_estimate("incomplete_response")
    retained: Final = logging_obj.baseline_cache_context
    assert retained is not None and retained.invalidated
    await logging_obj.invalidate_baseline_cache_estimate("retried_request")
    clock.now = 1200.0
    completed: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # retry completion observed later
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # retry's terminal native SSE event
        api_call_start_time=completed,
        completion_start_time=completed,
        prompt_cache_response_complete=True,
    )
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    clock.now = 4601.0
    following: Final = await _reservation(estimator, "after-partial-stream-start-ttl")
    estimate: Final = await estimator.finalize(
        following, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    assert estimate.status == "unknown"
    assert estimate.reason == "history_unavailable"
