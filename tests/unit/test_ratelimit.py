import pytest

from api.ratelimit import (
    DEGRADED_LOCAL_RATE_LIMIT,
    TOKEN_BUCKET_SCRIPT,
    LocalTokenBucketLimiter,
    RedisTokenBucketLimiter,
    build_rate_limiter,
)
from config.settings import RoleRateLimit, get_limits
from retrieval.models import UserContext

VIEWER = UserContext(user_id="vimal", role="viewer", allowed_access_levels=("public",))
ADMIN = UserContext(user_id="admin", role="admin", allowed_access_levels=("public",))
SMALL_LIMITS = {
    "viewer": RoleRateLimit(capacity=2, refill_per_minute=6),
    "admin": RoleRateLimit(capacity=5, refill_per_minute=6),
}


class FailingRedis:
    def register_script(self, script):
        async def fail(keys, args):
            raise ConnectionError("Simulated Redis outage")

        return fail


class FakeRedis:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def register_script(self, script):
        self.script = script

        async def run(keys, args):
            self.calls.append((keys, args))
            return self.reply

        return run


async def test_a_burst_is_allowed_and_then_refused_with_a_wait():
    limiter = LocalTokenBucketLimiter(SMALL_LIMITS)

    decisions = [await limiter.check(VIEWER) for _ in range(3)]

    assert [decision.allowed for decision in decisions] == [True, True, False]
    assert decisions[2].retry_after_s > 0


async def test_limits_are_per_user():
    # One busy user must not use up anybody else's requests.
    limiter = LocalTokenBucketLimiter(SMALL_LIMITS)
    other_viewer = UserContext(user_id="other", role="viewer", allowed_access_levels=("public",))

    for _ in range(3):
        await limiter.check(VIEWER)

    assert (await limiter.check(other_viewer)).allowed


async def test_thresholds_differ_by_role():
    limiter = LocalTokenBucketLimiter(SMALL_LIMITS)

    admin_decisions = [await limiter.check(ADMIN) for _ in range(5)]

    assert all(decision.allowed for decision in admin_decisions)


async def test_a_bucket_refills_over_time(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("api.ratelimit.time.monotonic", lambda: clock[0])
    limiter = LocalTokenBucketLimiter(SMALL_LIMITS)
    for _ in range(3):
        await limiter.check(VIEWER)

    clock[0] += 10.0

    assert (await limiter.check(VIEWER)).allowed


@pytest.mark.faults
async def test_a_redis_outage_falls_back_and_still_limits():
    limiter = RedisTokenBucketLimiter(
        FailingRedis(), SMALL_LIMITS, LocalTokenBucketLimiter(SMALL_LIMITS)
    )

    decisions = [await limiter.check(VIEWER) for _ in range(3)]

    assert [decision.allowed for decision in decisions] == [True, True, False]
    assert decisions[0].degraded == [DEGRADED_LOCAL_RATE_LIMIT]


@pytest.mark.faults
async def test_the_redis_fault_switch_uses_the_fallback():
    limiter = RedisTokenBucketLimiter(
        FakeRedis([1, "0"]),
        SMALL_LIMITS,
        LocalTokenBucketLimiter(SMALL_LIMITS),
        enabled_faults=("redis",),
    )

    assert (await limiter.check(VIEWER)).degraded == [DEGRADED_LOCAL_RATE_LIMIT]


async def test_redis_replies_become_decisions():
    limiter = RedisTokenBucketLimiter(
        FakeRedis([0, "12.5"]), SMALL_LIMITS, LocalTokenBucketLimiter(SMALL_LIMITS)
    )

    decision = await limiter.check(VIEWER)

    assert decision.allowed is False
    assert decision.retry_after_s == 13


def test_no_redis_address_uses_the_local_limiter():
    assert isinstance(build_rate_limiter("", SMALL_LIMITS), LocalTokenBucketLimiter)


def test_each_role_has_a_limit():
    assert set(get_limits().rate_limits) == {"viewer", "analyst", "admin"}


def test_the_script_spends_tokens_in_one_step():
    # The bucket is read and written inside one script, which Redis runs atomically.
    assert "HMGET" in TOKEN_BUCKET_SCRIPT and "HSET" in TOKEN_BUCKET_SCRIPT
