"""Circuit breaker: stop calling a service that keeps failing, and retry it after a pause.

Without this, every request would wait for a timeout from a service that is already down.
It knows nothing about any one service, so it can guard any dependency.
"""

import time
from collections.abc import Callable


class CircuitBreaker:
    """Tracks consecutive failures of one service. It is closed (healthy) until it opens."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set the limits. The clock can be replaced, so that tests need not really wait."""
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Return True while calls should be skipped."""
        if self._opened_at is None:
            return False
        return self._clock() - self._opened_at < self._recovery_seconds

    def allows_request(self) -> bool:
        """Return True if the service may be called.

        After the recovery pause this returns True for every caller, so concurrent requests
        may all try a service that is still down. This keeps the code simple, and the cost
        is small: the failure count is not reset, so the first failure reopens the breaker.
        """
        return not self.is_open

    def record_success(self) -> None:
        """Close the breaker, because the service answered."""
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Count a failure, and open the breaker when the threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = self._clock()
