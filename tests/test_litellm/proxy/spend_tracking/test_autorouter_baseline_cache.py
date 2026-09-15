import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from redis.exceptions import RedisError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimate,
    BaselineCacheEstimator,
    BaselineReservation,
)

_MODEL: Final = "claude-sonnet-5"
_TARGET: Final = NativePredictionTarget(_MODEL, "test-provider-key")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


@dataclass
class _Clock:
    now: float = 10000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Counter:
    def __init__(self, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.calls = 0

    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        self.calls += 1
        return None if self.unavailable else json.dumps(_JSON_OBJECT.validate_python(body)).count("token ")


def _wire(ttl: str = "1h", *, growth: int = 0, changed: bool = False) -> httpx.Request:
    grown: Final = f'{{"type":"text","text":"{"token " * growth}"}},' if growth else ""
    return httpx.Request(
        "POST",
        "https://configured-native-provider.test/v1/messages",
        headers=MappingProxyType({"anthropic-version": "2023-06-01"}),
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"{("changed " if changed else "") + "token " * 6000}"}},
                {grown}
                {{"type":"text","text":"end","cache_control":{{"type":"ephemeral","ttl":"{ttl}"}}}}
            ]}}]
        }}''',
    )


async def _reserve(
    estimator: BaselineCacheEstimator,
    request: str,
    *,
    caller: str = "caller",
    session: str = "session",
    target: NativePredictionTarget = _TARGET,
) -> BaselineReservation:
    reservation: Final = await estimator.reserve(
        caller_key_hash=caller,
        session_id=session,
        router_id="router",
        baseline_deployment_id="baseline",
        target=target,
        request_id=request,
    )
    assert isinstance(reservation, BaselineReservation)
    return reservation


async def _run(
    estimator: BaselineCacheEstimator,
    clock: _Clock,
    request: str,
    wire: httpx.Request,
) -> BaselineCacheEstimate:
    reservation: Final = await _reserve(estimator, request)
    started: Final = clock.now
    clock.advance(0.1)
    return await estimator.finalize(reservation, wire=wire, request_started_at=started, available_at=clock.now)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ttl,duration,bucket",
    (("5m", 300, "cache_creation_5m_input_tokens"), ("1h", 3600, "cache_creation_1h_input_tokens")),
)
async def test_established_expiry_never_receives_a_hypothetical_read(ttl: str, duration: int, bucket: str) -> None:
    clock: Final = _Clock()
    counter: Final = _Counter()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    first: Final = await _run(estimator, clock, "first", _wire(ttl))
    assert (first.status, first.reason) == ("unknown", "history_unavailable")
    clock.now = 10001.0
    warm: Final = await _run(estimator, clock, "warm", _wire(ttl))
    assert warm.cache_read_input_tokens == 6000
    assert warm.cache_creation_1h_input_tokens == warm.cache_creation_5m_input_tokens == 0
    clock.now = 10001.0 + duration
    expired: Final = await _run(estimator, clock, "expired", _wire(ttl))
    assert expired.status == "estimated"
    assert expired.reason == "cache_prefix_expired"
    assert expired.cache_read_input_tokens == 0
    assert expired.metadata()[bucket] == 6000
    usage: Final = expired.usage(10)
    assert usage is not None and usage.prompt_tokens == 6000 and usage.total_tokens == 6010
    assert counter.calls <= 4


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl,duration", (("5m", 300), ("1h", 3600)))
@pytest.mark.parametrize("cancelled", (False, True))
async def test_invalidation_keeps_unknown_cache_effects_through_latest_retry_completion(
    ttl: str, duration: int, cancelled: bool
) -> None:
    clock: Final = _Clock()
    counter: Final = _Counter()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    await _run(estimator, clock, "seed", _wire(ttl))
    clock.now = 10001.0
    warm: Final = await _run(estimator, clock, "warm", _wire(ttl))
    assert warm.cache_read_input_tokens == 6000
    clock.now = 10005.0
    failed: Final = await _reserve(estimator, "retrying")
    if cancelled:
        await estimator.cancel(failed)
    calls: Final = counter.calls
    clock.now = 10010.0
    abandoned: Final = await estimator.invalidate(failed, "upstream_request_failed")
    assert (abandoned.status, abandoned.reason) == ("unknown", "upstream_request_failed")
    clock.now = 10010.0 + duration
    retried: Final = await estimator.invalidate(failed, "retried_upstream_request")
    assert (retried.status, retried.reason) == ("unknown", "retried_upstream_request")
    assert counter.calls == calls
    clock.now = 10020.0 + duration
    following: Final = await _run(estimator, clock, "following", _wire(ttl))
    assert (following.status, following.reason) == ("unknown", "history_unavailable")
    assert following.cache_read_input_tokens is None
    clock.now = 10020.0 + 2 * duration
    expired: Final = await _run(estimator, clock, "expired", _wire(ttl))
    assert (expired.status, expired.reason) == ("estimated", "cache_prefix_expired")
    assert expired.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_prefix_change_is_cold_after_history_horizon_and_unknown_before_it() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire())
    early: Final = await _run(estimator, clock, "changed-early", _wire(changed=True))
    assert early.status == "unknown"
    clock.now = 13601.0
    changed: Final = await _run(estimator, clock, "changed", _wire(growth=2000))
    assert changed.status == "estimated"
    assert changed.cache_read_input_tokens == 0
    assert changed.cache_creation_1h_input_tokens == 8000


@pytest.mark.asyncio
async def test_lookback_reads_prior_marker_and_writes_only_growth() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire())
    clock.now = 13600.0
    await _run(estimator, clock, "cold", _wire())
    growing_wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"{"token " * 6000}"}},
                {{"type":"text","text":"end"}},
                {{"type":"text","text":"{"token " * 2000}","cache_control":{{"type":"ephemeral","ttl":"1h"}}}}
            ]}}]
        }}''',
    )
    grown: Final = await _run(estimator, clock, "grown", growing_wire)
    assert grown.status == "estimated"
    assert grown.cache_read_input_tokens == 6000
    assert grown.cache_creation_1h_input_tokens == 2000
    clock.now = 17200.05
    old_prefix: Final = await _run(estimator, clock, "old-prefix", _wire())
    assert old_prefix.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_mixed_ttl_keeps_new_hour_and_five_minute_writes_separate() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "seed", _wire())
    clock.now = 13600.0
    wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"{"token " * 5000}","cache_control":{{"type":"ephemeral","ttl":"1h"}}}},
                {{"type":"text","text":"{"token " * 2000}","cache_control":{{"type":"ephemeral","ttl":"5m"}}}},
                {{"type":"text","text":"{"token " * 100}"}}
            ]}}]
        }}''',
    )
    result: Final = await _run(estimator, clock, "mixed", wire)
    assert result.status == "estimated"
    assert result.input_tokens == 100
    assert result.cache_read_input_tokens == 0
    assert result.cache_creation_1h_input_tokens == 5000
    assert result.cache_creation_5m_input_tokens == 2000


@pytest.mark.asyncio
async def test_parallel_pending_requests_and_late_completions_do_not_self_hit_or_regress_refresh() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    first: Final = await _reserve(estimator, "first")
    clock.now = 10001.0
    second: Final = await _reserve(estimator, "second")
    clock.now = 10002.0
    second_result: Final = await estimator.finalize(
        second, wire=_wire(), request_started_at=10001.0, available_at=10002.0
    )
    assert second_result.reason == "pending_request"
    first_result: Final = await estimator.finalize(
        first, wire=_wire(), request_started_at=10000.0, available_at=10002.0
    )
    assert first_result.cache_read_input_tokens is None
    duplicate: Final = await estimator.finalize(
        second, wire=_wire(changed=True), request_started_at=10001.0, available_at=10002.0
    )
    assert duplicate == second_result
    clock.now = 13600.5
    still_warm: Final = await _run(estimator, clock, "still-warm", _wire())
    assert still_warm.cache_read_input_tokens == 6000


@pytest.mark.asyncio
@pytest.mark.parametrize("complete,cache_hit", ((False, False), (True, True)))
async def test_failed_incomplete_or_gateway_cache_responses_do_not_warm_state(complete: bool, cache_hit: bool) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    reservation: Final = await _reserve(estimator, "ignored")
    result: Final = await estimator.finalize(
        reservation,
        wire=_wire(),
        request_started_at=clock.now,
        available_at=clock.now,
        completed=complete,
        cache_hit=cache_hit,
    )
    assert result.status == "unknown"
    following: Final = await _run(estimator, clock, "following", _wire())
    assert following.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_provider_counts_are_memoized_and_failures_remain_unknown() -> None:
    clock: Final = _Clock()
    counter: Final = _Counter(unavailable=True)
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    unavailable: Final = await _run(estimator, clock, "unavailable", _wire())
    assert unavailable.reason == "token_count_unavailable"
    assert unavailable.usage(5) is None
    counter.unavailable = False
    first: Final = await _run(estimator, clock, "first", _wire())
    assert first.reason == "history_unavailable"
    calls: Final = counter.calls
    second: Final = await _run(estimator, clock, "second", _wire())
    assert second.cache_read_input_tokens == 6000
    assert counter.calls == calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,session,target",
    (
        ("other", "session", _TARGET),
        ("caller", "other", _TARGET),
        ("caller", "session", NativePredictionTarget(_MODEL, "other-key")),
        ("caller", "session", NativePredictionTarget(_MODEL, "test-provider-key", "https://other.test")),
    ),
)
async def test_scope_isolates_callers_sessions_and_baseline_credentials(
    caller: str, session: str, target: NativePredictionTarget
) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire())
    isolated: Final = await _reserve(estimator, "second", caller=caller, session=session, target=target)
    result: Final = await estimator.finalize(
        isolated, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert result.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_uncacheable_prompt_does_not_need_history() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"token ","cache_control":{{"type":"ephemeral","ttl":"1h"}}}}
            ]}}]
        }}''',
    )
    result: Final = await _run(estimator, clock, "short", wire)
    assert result.reason == "below_cache_minimum"
    assert result.input_tokens == 1
    assert result.cache_read_input_tokens == result.cache_creation_1h_input_tokens == 0


@pytest.mark.asyncio
async def test_ttl_changes_remain_unknown_until_every_possible_refreshed_entry_expires() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire("1h"))
    warm: Final = await _run(estimator, clock, "warm", _wire("1h"))
    assert warm.cache_read_input_tokens == 6000
    clock.now = 10002.0
    changed: Final = await _run(estimator, clock, "changed", _wire("5m"))
    assert (changed.status, changed.reason) == ("unknown", "cache_ttl_changed")
    clock.now = 10303.0
    ambiguous: Final = await _run(estimator, clock, "ambiguous", _wire("5m"))
    assert ambiguous.reason == "cache_ttl_changed"
    clock.now = 13602.5
    refreshed: Final = await _run(estimator, clock, "refreshed", _wire("5m"))
    assert refreshed.reason == "cache_ttl_changed"
    clock.now = 17204.0
    expired: Final = await _run(estimator, clock, "expired", _wire("5m"))
    assert expired.status == "estimated"
    assert expired.cache_read_input_tokens == 0
    assert expired.cache_creation_5m_input_tokens == 6000


@pytest.mark.asyncio
async def test_unmarked_provider_cache_activity_is_unknown_but_plain_usage_is_estimable() -> None:
    clock: Final = _Clock()
    counter: Final = _Counter()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":"{"token " * 6000}"}}]
        }}''',
    )
    reservation: Final = await _reserve(estimator, "implicit")
    implicit: Final = await estimator.finalize(
        reservation,
        wire=wire,
        request_started_at=clock.now,
        available_at=clock.now,
        observed_cache_tokens=6000,
    )
    assert implicit.reason == "implicit_cache_without_breakpoints"
    assert counter.calls == 0
    ordinary: Final = await _run(estimator, clock, "ordinary", wire)
    assert ordinary.status == "estimated" and ordinary.input_tokens == 6000
    assert ordinary.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_pruned_pending_reservation_cannot_advance_cache_state() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    first: Final = await _reserve(estimator, "first")
    for index in range(256):
        await _reserve(estimator, f"pending-{index}")
    pruned: Final = await estimator.finalize(
        first,
        wire=_wire(),
        request_started_at=clock.now,
        available_at=clock.now,
    )
    assert pruned.reason == "reservation_unavailable"


@pytest.mark.asyncio
async def test_long_running_request_keeps_the_cache_history_needed_at_its_start() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "seed", _wire("5m"))
    clock.now = 10001.0
    delayed: Final = await _reserve(estimator, "delayed")
    clock.now = 15000.0
    result: Final = await estimator.finalize(
        delayed,
        wire=_wire("5m"),
        request_started_at=10001.0,
        available_at=15000.0,
    )
    assert result.status == "estimated"
    assert result.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_redis_is_authoritative_and_faults_never_fall_back_to_local_warmth() -> None:
    clock: Final = _Clock()
    backend: Final = RedisCache(host="127.0.0.1", port=6398, namespace="baseline-state-test")
    cache: Final = DualCache(redis_cache=backend)
    first_process: Final = BaselineCacheEstimator(cache, clock, _Counter())
    second_process: Final = BaselineCacheEstimator(cache, clock, _Counter())
    try:
        await backend.async_delete_cache("unused")
    except (RedisError, OSError):
        pytest.skip("isolated integration Redis is unavailable")
    session: Final = str(id(first_process))
    first: Final = await _reserve(first_process, "first", session=session)
    clock.now += 0.1
    await first_process.finalize(first, wire=_wire(), request_started_at=first.reserved_at, available_at=clock.now)
    second: Final = await _reserve(second_process, "second", session=session)
    warm: Final = await second_process.finalize(
        second, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert warm.cache_read_input_tokens == 6000
    await backend.async_delete_cache(first.scope)
    lost: Final = await _reserve(first_process, "lost", session=session)
    unknown: Final = await first_process.finalize(
        lost, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert unknown.reason == "history_unavailable"
    broken_cache: Final = DualCache(redis_cache=RedisCache(host="127.0.0.1", port=1))
    broken: Final = BaselineCacheEstimator(broken_cache, clock, _Counter())
    refused: Final = await broken.reserve(
        caller_key_hash="caller",
        session_id=session,
        router_id="router",
        baseline_deployment_id="baseline",
        target=_TARGET,
        request_id="broken",
    )
    assert isinstance(refused, BaselineCacheEstimate) and refused.reason == "state_unavailable"
