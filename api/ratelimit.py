"""Per-user rate limiting with a token bucket, checked before a turn starts.

Each user has a bucket that refills steadily, so a short burst is fine and a sustained flood is
not. Redis holds the buckets, so that limits survive a restart and apply across API processes. The
check is a Lua script, because Redis runs a script as one step, and two requests at once cannot
both spend the last token.

If Redis fails, an in-process bucket takes over. It is per process, but it keeps the limit
working, and rate limiting never becomes a reason to refuse an employee (failure table).
"""

import math
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Protocol

import redis.asyncio as redis
import structlog

from config.settings import RoleRateLimit
from retrieval.models import UserContext

logger = structlog.get_logger()

REDIS_FAULT = "redis"
DEGRADED_LOCAL_RATE_LIMIT = "rate_limit_in_process"
BUCKET_KEY_TEMPLATE = "rate_limit:{user_id}"
SECONDS_PER_MINUTE = 60
# Idle buckets expire, so that Redis does not keep one per user forever.
BUCKET_EXPIRY_SECONDS = 3600
REDIS_TIMEOUT_S = 0.5

# Takes the bucket's key, capacity and refill per second. Returns [allowed, seconds to wait].
# Time comes from Redis, so that API processes with different clocks agree.
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_second = tonumber(ARGV[2])
local expiry_seconds = tonumber(ARGV[3])

local redis_time = redis.call('TIME')
local now = tonumber(redis_time[1]) + tonumber(redis_time[2]) / 1000000

local bucket = redis.call('HMGET', key, 'tokens', 'updated_at')
local tokens = tonumber(bucket[1]) or capacity
local updated_at = tonumber(bucket[2]) or now

tokens = math.min(capacity, tokens + (now - updated_at) * refill_per_second)

local allowed = 0
local retry_after = 0
if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
else
    retry_after = (1 - tokens) / refill_per_second
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'updated_at', tostring(now))
redis.call('EXPIRE', key, expiry_seconds)
return {allowed, tostring(retry_after)}
"""


@dataclass(frozen=True)
class RateDecision:
    """Whether a request may go ahead."""

    allowed: bool
    # Whole seconds to wait, for the `Retry-After` header.
    retry_after_s: int = 0
    # Set when the fallback decided, so that the UI can show it.
    degraded: list[str] = field(default_factory=list)


class RateLimiter(Protocol):
    """What the chat route needs, so that tests can run with no Redis."""

    async def check(self, user: UserContext) -> RateDecision:
        """Spend one token for this user, if one is available."""
        ...


class LocalTokenBucketLimiter:
    """Token buckets in this process, for when Redis is unavailable."""

    def __init__(self, limits_by_role: dict[str, RoleRateLimit]) -> None:
        """Start with no buckets, as each one is created full on its first use."""
        self._limits_by_role = limits_by_role
        self._buckets: dict[str, tuple[float, float]] = {}

    async def check(self, user: UserContext) -> RateDecision:
        """Refill the user's bucket for the time passed, and spend one token."""
        limit = self._limits_by_role[user.role]
        refill_per_second = limit.refill_per_minute / SECONDS_PER_MINUTE
        now = time.monotonic()
        tokens, updated_at = self._buckets.get(user.user_id, (float(limit.capacity), now))
        tokens = min(limit.capacity, tokens + (now - updated_at) * refill_per_second)

        if tokens >= 1:
            self._buckets[user.user_id] = (tokens - 1, now)
            return RateDecision(allowed=True)
        self._buckets[user.user_id] = (tokens, now)
        return RateDecision(
            allowed=False, retry_after_s=math.ceil((1 - tokens) / refill_per_second)
        )


class RedisTokenBucketLimiter:
    """Token buckets in Redis, with the local limiter as the fallback."""

    def __init__(
        self,
        redis_client: Any,
        limits_by_role: dict[str, RoleRateLimit],
        fallback: LocalTokenBucketLimiter,
        enabled_faults: Collection[str] = (),
    ) -> None:
        """Keep the client. The script is registered once, and Redis caches it."""
        self._limits_by_role = limits_by_role
        self._fallback = fallback
        self._enabled_faults = enabled_faults
        self._spend_token = redis_client.register_script(TOKEN_BUCKET_SCRIPT)

    async def check(self, user: UserContext) -> RateDecision:
        """Spend one token in Redis, or in this process if Redis cannot answer."""
        limit = self._limits_by_role[user.role]
        try:
            if REDIS_FAULT in self._enabled_faults:
                raise ConnectionError("Simulated Redis failure from FAULTS")
            allowed, retry_after = await self._spend_token(
                keys=[BUCKET_KEY_TEMPLATE.format(user_id=user.user_id)],
                args=[
                    limit.capacity,
                    limit.refill_per_minute / SECONDS_PER_MINUTE,
                    BUCKET_EXPIRY_SECONDS,
                ],
            )
        except Exception as error:  # Any Redis failure leads to the same fallback.
            logger.warning("rate_limit_redis_failed", error=type(error).__name__)
            decision = await self._fallback.check(user)
            return RateDecision(
                decision.allowed, decision.retry_after_s, [DEGRADED_LOCAL_RATE_LIMIT]
            )

        if allowed:
            return RateDecision(allowed=True)
        return RateDecision(allowed=False, retry_after_s=math.ceil(float(retry_after)))


def build_rate_limiter(
    redis_url: str,
    limits_by_role: dict[str, RoleRateLimit],
    enabled_faults: Collection[str] = (),
) -> RateLimiter:
    """Return the Redis limiter, or the local one if no Redis address is set."""
    local_limiter = LocalTokenBucketLimiter(limits_by_role)
    if not redis_url:
        return local_limiter
    redis_client = redis.from_url(
        redis_url,
        decode_responses=True,
        socket_timeout=REDIS_TIMEOUT_S,
        socket_connect_timeout=REDIS_TIMEOUT_S,
    )
    return RedisTokenBucketLimiter(redis_client, limits_by_role, local_limiter, enabled_faults)
