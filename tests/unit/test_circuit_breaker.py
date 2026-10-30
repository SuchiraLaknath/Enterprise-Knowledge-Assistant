from resilience.circuit_breaker import CircuitBreaker


class ManualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_breaker(clock: ManualClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, recovery_seconds=30.0, clock=clock)


def test_breaker_stays_closed_below_the_threshold():
    breaker = make_breaker(ManualClock())

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.allows_request()


def test_breaker_opens_at_the_threshold():
    breaker = make_breaker(ManualClock())

    for _ in range(3):
        breaker.record_failure()

    assert not breaker.allows_request()


def test_a_success_resets_the_failure_count():
    breaker = make_breaker(ManualClock())

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.allows_request()


def test_breaker_allows_a_trial_request_after_the_recovery_pause():
    clock = ManualClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()

    clock.now += 31

    assert breaker.allows_request()


def test_a_failed_trial_request_reopens_the_breaker():
    clock = ManualClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()
    clock.now += 31

    breaker.record_failure()

    assert not breaker.allows_request()


def test_a_successful_trial_request_closes_the_breaker():
    clock = ManualClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()
    clock.now += 31

    breaker.record_success()
    breaker.record_failure()

    assert breaker.allows_request()
