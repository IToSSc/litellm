from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.prompt_cache_prediction import (
    CountedBreakpoint,
    CountedPromptCachePlan,
    NativePredictionTarget,
    TokenCounter,
    UnsupportedCachePlan,
    count_cache_plan,
    count_prompt_tokens,
    parse_cache_plan,
    supported_prediction_headers,
)
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage
from litellm.utils import get_prompt_cache_min_tokens

_MAX_TTL: Final = 3600
_RETENTION_SECONDS: Final = 86400
_MAX_REQUESTS: Final = 256
_MAX_VERSIONS: Final = 1024
_MAX_SCOPES: Final = 1024
_MAX_COUNTS: Final = 4096
_COUNT_TIMEOUT: Final = 3.0
_STORE_TIMEOUT: Final = 1.0
_CAS_ATTEMPTS: Final = 4
_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])
_GET_SCRIPT: Final = "return redis.call('GET', KEYS[1])"
_CAS_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
if (current or '') ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
"""


class BaselineCacheEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    status: Literal["estimated", "unknown"]
    reason: str
    input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_5m_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_1h_input_tokens: int | None = Field(default=None, ge=0)

    def usage(self, completion_tokens: int) -> Usage | None:
        if (
            self.status != "estimated"
            or self.input_tokens is None
            or self.cache_read_input_tokens is None
            or self.cache_creation_5m_input_tokens is None
            or self.cache_creation_1h_input_tokens is None
        ):
            return None
        writes: Final = self.cache_creation_5m_input_tokens + self.cache_creation_1h_input_tokens
        total: Final = self.input_tokens + self.cache_read_input_tokens + writes
        return Usage(
            prompt_tokens=total,
            completion_tokens=completion_tokens,
            total_tokens=total + completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=self.input_tokens,
                cached_tokens=self.cache_read_input_tokens,
                cache_creation_tokens=writes,
                cache_write_tokens=writes,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=self.cache_creation_5m_input_tokens,
                    ephemeral_1h_input_tokens=self.cache_creation_1h_input_tokens,
                ),
            ),
        )

    def metadata(self) -> dict[str, JsonValue]:  # mutable-ok: spend-log JSON serializers require a plain dictionary
        return _JSON_BODY.validate_python(self.model_dump(mode="json", exclude_none=True))


def unknown_estimate(reason: str) -> BaselineCacheEstimate:
    return BaselineCacheEstimate(status="unknown", reason=reason)


@dataclass(frozen=True, slots=True)
class BaselineReservation:
    scope: str
    request_id: str
    reserved_at: float
    target: NativePredictionTarget


class _Pending(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str
    reserved_at: float


class _Completed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str
    completed_at: float
    estimate: BaselineCacheEstimate


class _Version(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fingerprint: str
    content_fingerprint: str
    prefix_tokens: int = Field(ge=0)
    ttl_seconds: Literal[300, 3600]
    started_at: float
    available_at: float
    expires_at: float
    uncertain: bool = False


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    uncertain_before: float
    pending: tuple[_Pending, ...] = ()
    completed: tuple[_Completed, ...] = ()
    versions: tuple[_Version, ...] = ()


@dataclass(frozen=True, slots=True)
class _Snapshot:
    raw: str
    state: _State | None


@dataclass(frozen=True, slots=True)
class _StoreFailure:
    reason: str = "state_unavailable"


@runtime_checkable
class _Script(Protocol):
    def __call__(self, *, keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> Awaitable[object]: ...


def _script(value: object) -> _Script | None:
    return value if isinstance(value, _Script) else None


class _HistoryStore:
    def __init__(self, cache: DualCache) -> None:
        self.cache = cache
        self.local: Mapping[str, tuple[float, str]] = MappingProxyType({})
        self.lock = asyncio.Lock()

    async def read(self, scope: str, now: float) -> _Snapshot | _StoreFailure:
        if self.cache.redis_cache is None:
            local_value: Final = self.local.get(scope)
            raw: Final = local_value[1] if local_value is not None and local_value[0] > now else None
            return self._decode(raw)
        try:
            script: Final = _script(self.cache.redis_cache.async_register_script(_GET_SCRIPT))
            if script is None:
                return _StoreFailure()
            value: Final[object] = await asyncio.wait_for(script(keys=(scope,), args=()), timeout=_STORE_TIMEOUT)
            return self._decode(value)
        except Exception:  # noqa: BLE001  # storage faults must not establish cache absence
            return _StoreFailure()

    @staticmethod
    def _decode(raw: object) -> _Snapshot | _StoreFailure:
        if raw is None:
            return _Snapshot(raw="", state=None)
        if not isinstance(raw, (str, bytes)):
            return _StoreFailure()
        try:
            text: Final = raw.decode() if isinstance(raw, bytes) else raw
            return _Snapshot(raw=text, state=_State.model_validate_json(text))
        except (UnicodeDecodeError, ValidationError):
            return _StoreFailure()

    async def exchange(self, scope: str, before: _Snapshot, after: _State, now: float) -> bool | _StoreFailure:
        serialized: Final = after.model_dump_json()
        if self.cache.redis_cache is not None:
            try:
                script: Final = _script(self.cache.redis_cache.async_register_script(_CAS_SCRIPT))
                if script is None:
                    return _StoreFailure()
                result: Final[object] = await asyncio.wait_for(
                    script(keys=(scope,), args=(before.raw, serialized, _RETENTION_SECONDS)),
                    timeout=_STORE_TIMEOUT,
                )
            except Exception:  # noqa: BLE001  # preserve uncertainty when Redis rejects an atomic update
                return _StoreFailure()
            return result == 1
        async with self.lock:
            existing: Final = self.local.get(scope)
            current: Final = existing[1] if existing is not None and existing[0] > now else ""
            if current != before.raw:
                return False
            remaining: Final = tuple(
                (key, entry) for key, entry in self.local.items() if key != scope and entry[0] > now
            )[-(_MAX_SCOPES - 1) :]
            self.local = MappingProxyType(
                {key: entry for key, entry in (*remaining, (scope, (now + _RETENTION_SECONDS, serialized)))}
            )
            return True


@dataclass(frozen=True, slots=True)
class _Count:
    tokens: int
    expires_at: float


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _key(caller: str, session: str, router: str, deployment: str, target: NativePredictionTarget) -> str:
    return "autorouter-baseline-cache:v1:" + _digest(
        (caller, session, router, deployment, target.model, target.api_key, target.api_base)
    )


def _bounded(state: _State, now: float) -> _State:
    stale_pending: Final = tuple(item for item in state.pending if item.reserved_at < now - 2 * _MAX_TTL)
    pending: Final = tuple(item for item in state.pending if item.reserved_at >= now - 2 * _MAX_TTL)
    needed_since: Final = min((now - _MAX_TTL, *(item.reserved_at for item in pending)))
    live_versions: Final = tuple(version for version in state.versions if version.expires_at >= needed_since)
    dropped_versions: Final = live_versions[:-_MAX_VERSIONS]
    dropped_pending: Final = pending[:-_MAX_REQUESTS]
    uncertainty: Final = max(
        (
            state.uncertain_before,
            *(version.started_at for version in dropped_versions),
            now if stale_pending or dropped_pending else 0.0,
        )
    )
    return _State(
        uncertain_before=uncertainty,
        pending=pending[-_MAX_REQUESTS:],
        completed=state.completed[-_MAX_REQUESTS:],
        versions=live_versions[-_MAX_VERSIONS:],
    )


def _eligible(plan: CountedPromptCachePlan, minimum: int) -> tuple[CountedBreakpoint, ...]:
    return tuple(marker for marker in plan.breakpoints if marker.prefix_tokens >= minimum)


def _matching(state: _State, markers: tuple[CountedBreakpoint, ...], started: float) -> tuple[_Version, ...]:
    return tuple(
        version
        for version in state.versions
        if not version.uncertain
        and version.available_at <= started < version.expires_at
        and any(
            version.fingerprint in marker.lookback_fingerprints
            and version.prefix_tokens <= marker.prefix_tokens
            and version.ttl_seconds == marker.ttl_seconds
            for marker in markers
        )
    )


def _ambiguous(state: _State, markers: tuple[CountedBreakpoint, ...], started: float) -> tuple[_Version, ...]:
    return tuple(
        version
        for version in state.versions
        if version.available_at <= started < version.expires_at
        and any(
            version.content_fingerprint in marker.lookback_content_fingerprints
            and (version.uncertain or version.ttl_seconds != marker.ttl_seconds)
            for marker in markers
        )
    )


def _estimate(
    state: _State, request_id: str, plan: CountedPromptCachePlan, minimum: int, started: float
) -> BaselineCacheEstimate:
    markers: Final = _eligible(plan, minimum)
    if not markers:
        return BaselineCacheEstimate(
            status="estimated",
            reason="below_cache_minimum" if plan.breakpoints else "no_cache_breakpoints",
            input_tokens=plan.total_tokens,
            cache_read_input_tokens=0,
            cache_creation_5m_input_tokens=0,
            cache_creation_1h_input_tokens=0,
        )
    if any(item.request_id != request_id and item.reserved_at <= started for item in state.pending):
        return unknown_estimate("pending_request")
    if _ambiguous(state, markers, started):
        return unknown_estimate("cache_ttl_changed")
    candidates: Final = _matching(state, markers, started)
    read: Final = max((version.prefix_tokens for version in candidates), default=0)
    end: Final = markers[-1].prefix_tokens
    if read < end and started < state.uncertain_before + max(marker.ttl_seconds for marker in markers):
        return unknown_estimate("history_unavailable")
    one_hour: Final = max(
        (marker.prefix_tokens for marker in markers if marker.ttl_seconds == 3600 and marker.prefix_tokens > read),
        default=read,
    )
    expired: Final = any(
        version.expires_at <= started and any(version.fingerprint in marker.lookback_fingerprints for marker in markers)
        for version in state.versions
    )
    return BaselineCacheEstimate(
        status="estimated",
        reason="cache_prefix_available" if read else "cache_prefix_expired" if expired else "cache_prefix_cold",
        input_tokens=plan.total_tokens - end,
        cache_read_input_tokens=read,
        cache_creation_5m_input_tokens=end - one_hour,
        cache_creation_1h_input_tokens=one_hour - read,
    )


def _new_versions(
    state: _State,
    markers: tuple[CountedBreakpoint, ...],
    started: float,
    available: float,
) -> tuple[_Version, ...]:
    ambiguous: Final = _ambiguous(state, markers, started)
    if ambiguous:
        longest_ttl: Final = max(version.expires_at - version.started_at for version in ambiguous)
        return tuple(
            _Version(
                fingerprint=marker.fingerprint,
                content_fingerprint=marker.content_fingerprint,
                prefix_tokens=marker.prefix_tokens,
                ttl_seconds=3600 if marker.ttl_seconds == 3600 else 300,
                started_at=started,
                available_at=available,
                expires_at=started + max(longest_ttl, marker.ttl_seconds),
                uncertain=True,
            )
            for marker in markers
        )
    candidates: Final = _matching(state, markers, started)
    hit: Final = max(candidates, key=lambda version: version.prefix_tokens, default=None)
    refresh: Final = (
        (
            _Version(
                fingerprint=hit.fingerprint,
                content_fingerprint=hit.content_fingerprint,
                prefix_tokens=hit.prefix_tokens,
                ttl_seconds=hit.ttl_seconds,
                started_at=started,
                available_at=available,
                expires_at=started + hit.ttl_seconds,
            ),
        )
        if hit is not None and all(marker.fingerprint != hit.fingerprint for marker in markers)
        else ()
    )
    return (
        *refresh,
        *(
            _Version(
                fingerprint=marker.fingerprint,
                content_fingerprint=marker.content_fingerprint,
                prefix_tokens=marker.prefix_tokens,
                ttl_seconds=3600 if marker.ttl_seconds == 3600 else 300,
                started_at=started,
                available_at=available,
                expires_at=started + marker.ttl_seconds,
            )
            for marker in markers
        ),
    )


class BaselineCacheEstimator:
    def __init__(
        self,
        cache: DualCache,
        clock: Callable[[], float] = time.time,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = _HistoryStore(cache)
        self.clock = clock
        self.token_counter = token_counter
        self.counts: Mapping[str, _Count] = MappingProxyType({})
        self.count_slots = asyncio.Semaphore(8)
        self.uncertainty_debt: Mapping[str, float] = MappingProxyType({})
        self.uncertainty_floor = 0.0

    def _storage_failed(self, scope: str, now: float, reason: str = "state_unavailable") -> BaselineCacheEstimate:
        outstanding: Final = tuple(
            (key, stamp) for key, stamp in self.uncertainty_debt.items() if key != scope and stamp + _MAX_TTL > now
        )
        self.uncertainty_floor = max(
            (self.uncertainty_floor, *(stamp for _, stamp in outstanding[: -(_MAX_SCOPES - 1)]))
        )
        remaining: Final = outstanding[-(_MAX_SCOPES - 1) :]
        self.uncertainty_debt = MappingProxyType({key: stamp for key, stamp in (*remaining, (scope, now))})
        return unknown_estimate(reason)

    async def reserve(
        self,
        *,
        caller_key_hash: str,
        session_id: str,
        router_id: str,
        baseline_deployment_id: str,
        target: NativePredictionTarget,
        request_id: str,
    ) -> BaselineReservation | BaselineCacheEstimate:
        if not all((caller_key_hash, session_id, router_id, baseline_deployment_id, request_id)):
            return unknown_estimate("missing_baseline_scope")
        now: Final = self.clock()
        scope: Final = _key(caller_key_hash, session_id, router_id, baseline_deployment_id, target)
        record_id: Final = _digest(request_id)

        async def attempt(remaining: int) -> BaselineReservation | BaselineCacheEstimate:
            before: Final = await self.store.read(scope, now)
            if isinstance(before, _StoreFailure):
                return self._storage_failed(scope, now, before.reason)
            loaded: Final = before.state or _State(uncertain_before=now)
            state: Final = _bounded(
                loaded.model_copy(
                    update=MappingProxyType(
                        {
                            "uncertain_before": max(
                                loaded.uncertain_before, self.uncertainty_debt.get(scope, 0.0), self.uncertainty_floor
                            )
                        }
                    )
                ),
                now,
            )
            prior: Final = next((item for item in state.pending if item.request_id == record_id), None)
            if prior is not None:
                return BaselineReservation(scope, record_id, prior.reserved_at, target)
            finished: Final = next((item for item in state.completed if item.request_id == record_id), None)
            if finished is not None:
                return finished.estimate
            after: Final = _bounded(
                state.model_copy(
                    update=MappingProxyType(
                        {"pending": (*state.pending, _Pending(request_id=record_id, reserved_at=now))}
                    )
                ),
                now,
            )
            exchanged: Final = await self.store.exchange(scope, before, after, now)
            if isinstance(exchanged, _StoreFailure):
                return self._storage_failed(scope, now, exchanged.reason)
            if exchanged:
                return BaselineReservation(scope, record_id, now, target)
            return (
                await attempt(remaining - 1) if remaining > 1 else self._storage_failed(scope, now, "state_contention")
            )

        return await attempt(_CAS_ATTEMPTS)

    async def _count(self, target: NativePredictionTarget, body: Mapping[str, JsonValue]) -> int | None:
        cache_key: Final = _digest((target.model, target.api_key, target.api_base, _JSON_BODY.validate_python(body)))
        now: Final = self.clock()
        cached: Final = self.counts.get(cache_key)
        if cached is not None and cached.expires_at > now:
            return cached.tokens

        async def execute() -> int | None:
            async with self.count_slots:
                return (
                    await self.token_counter(target.model, target.api_key, body)
                    if self.token_counter is not None
                    else await count_prompt_tokens(target.model, target.api_key, body, api_base=target.api_base)
                )

        try:
            tokens: Final = await asyncio.wait_for(execute(), timeout=_COUNT_TIMEOUT)
        except Exception:  # noqa: BLE001  # provider counting cannot fail a completed generation
            return None
        if tokens is None or tokens < 0:
            return None
        retained: Final = tuple(
            (key, value) for key, value in self.counts.items() if value.expires_at > now and key != cache_key
        )[-(_MAX_COUNTS - 1) :]
        self.counts = MappingProxyType(
            {key: value for key, value in (*retained, (cache_key, _Count(tokens, now + _MAX_TTL)))}
        )
        return tokens

    async def finalize(
        self,
        reservation: BaselineReservation,
        *,
        wire: httpx.Request,
        request_started_at: float,
        available_at: float,
        completed: bool = True,
        cache_hit: bool = False,
        observed_cache_tokens: int = 0,
    ) -> BaselineCacheEstimate:
        if cache_hit:
            await self.cancel(reservation)
            return unknown_estimate("response_cache_hit")
        if not completed:
            return await self.invalidate(reservation, "incomplete_response")
        if not reservation.reserved_at <= request_started_at <= available_at <= self.clock():
            return await self._finish(reservation, None, request_started_at, available_at, "invalid_request_timing")
        try:
            body: Final = _JSON_BODY.validate_json(wire.content)
        except (ValidationError, RuntimeError, httpx.RequestNotRead):
            return await self._finish(reservation, None, request_started_at, available_at, "invalid_wire_request")
        if not supported_prediction_headers(wire.headers):
            return await self._finish(
                reservation, None, request_started_at, available_at, "unsupported_request_headers"
            )
        plan: Final = parse_cache_plan(body)
        if isinstance(plan, UnsupportedCachePlan):
            return await self._finish(reservation, None, request_started_at, available_at, plan.reason)
        if not plan.breakpoints and observed_cache_tokens > 0:
            return await self._finish(
                reservation, None, request_started_at, available_at, "implicit_cache_without_breakpoints"
            )

        async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
            return await self._count(reservation.target, body)

        try:
            counted: Final = await asyncio.wait_for(
                count_cache_plan(reservation.target.model, reservation.target.api_key, plan, token_counter=count),
                timeout=_COUNT_TIMEOUT,
            )
        except TimeoutError:
            return await self._finish(reservation, None, request_started_at, available_at, "token_count_timeout")
        return await self._finish(
            reservation,
            None if isinstance(counted, UnsupportedCachePlan) else counted,
            request_started_at,
            available_at,
            counted.reason if isinstance(counted, UnsupportedCachePlan) else None,
        )

    async def _finish(
        self,
        reservation: BaselineReservation,
        plan: CountedPromptCachePlan | None,
        started: float,
        available: float,
        reason: str | None,
        *,
        invalidated: bool = False,
    ) -> BaselineCacheEstimate:
        now: Final = self.clock()
        minimum: Final = get_prompt_cache_min_tokens(reservation.target.model)

        async def attempt(remaining: int) -> BaselineCacheEstimate:
            before: Final = await self.store.read(reservation.scope, now)
            if isinstance(before, _StoreFailure):
                return self._storage_failed(reservation.scope, now, before.reason)
            if before.state is None and not invalidated:
                return unknown_estimate("history_unavailable")
            loaded: Final = before.state or _State(uncertain_before=now)
            state: Final = _bounded(
                loaded.model_copy(
                    update=MappingProxyType(
                        {
                            "uncertain_before": max(
                                loaded.uncertain_before,
                                self.uncertainty_debt.get(reservation.scope, 0.0),
                                self.uncertainty_floor,
                            )
                        }
                    )
                ),
                now,
            )
            prior: Final = next((item for item in state.completed if item.request_id == reservation.request_id), None)
            if prior is not None and not invalidated:
                return prior.estimate
            if not invalidated and not any(item.request_id == reservation.request_id for item in state.pending):
                return self._storage_failed(reservation.scope, now, "reservation_unavailable")
            estimate: Final = (
                _estimate(state, reservation.request_id, plan, minimum, started)
                if plan is not None
                else unknown_estimate(reason or "unsupported_request")
            )
            versions: Final = (
                _new_versions(state, _eligible(plan, minimum), started, available) if plan is not None else ()
            )
            after: Final = _bounded(
                _State(
                    uncertain_before=max(state.uncertain_before, started) if plan is None else state.uncertain_before,
                    pending=tuple(item for item in state.pending if item.request_id != reservation.request_id),
                    completed=(
                        *(item for item in state.completed if item.request_id != reservation.request_id),
                        _Completed(request_id=reservation.request_id, completed_at=now, estimate=estimate),
                    ),
                    versions=(*state.versions, *versions),
                ),
                now,
            )
            exchanged: Final = await self.store.exchange(reservation.scope, before, after, now)
            if isinstance(exchanged, _StoreFailure):
                return self._storage_failed(reservation.scope, now, exchanged.reason)
            if exchanged:
                return estimate
            return (
                await attempt(remaining - 1)
                if remaining > 1
                else self._storage_failed(reservation.scope, now, "state_contention")
            )

        return await attempt(_CAS_ATTEMPTS)

    async def invalidate(self, reservation: BaselineReservation, reason: str) -> BaselineCacheEstimate:
        now: Final = self.clock()
        return await self._finish(reservation, None, now, now, reason, invalidated=True)

    async def cancel(self, reservation: BaselineReservation) -> None:
        now: Final = self.clock()

        async def attempt(remaining: int) -> None:
            before: Final = await self.store.read(reservation.scope, now)
            if isinstance(before, _StoreFailure) or before.state is None:
                return
            state: Final = before.state
            after: Final = state.model_copy(
                update=MappingProxyType(
                    {"pending": tuple(item for item in state.pending if item.request_id != reservation.request_id)}
                )
            )
            exchanged: Final = await self.store.exchange(reservation.scope, before, after, now)
            if isinstance(exchanged, _StoreFailure) or exchanged:
                return
            if remaining > 1:
                await attempt(remaining - 1)

        await attempt(_CAS_ATTEMPTS)
