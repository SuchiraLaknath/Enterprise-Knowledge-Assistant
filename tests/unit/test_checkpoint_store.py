import pytest

from api.checkpoint_store import PostgresCheckpointStore


class FakeConnection:
    async def execute(self, query):
        return None


class FakeConnectionContext:
    def __init__(self, fails):
        self.fails = fails

    async def __aenter__(self):
        if self.fails:
            raise ConnectionError("Simulated Postgres outage")
        return FakeConnection()

    async def __aexit__(self, *details):
        return False


class FakePool:
    def __init__(self, fails=False):
        self.fails = fails

    def connection(self):
        return FakeConnectionContext(self.fails)


async def test_an_available_database_sets_up_once():
    setups = []

    async def record_setup():
        setups.append(1)

    store = PostgresCheckpointStore(FakePool(), [record_setup], timeout_s=1.0)

    assert await store.is_available()
    assert await store.is_available()
    assert setups == [1]


@pytest.mark.faults
async def test_a_failing_database_is_reported_and_recovers_by_itself():
    pool = FakePool(fails=True)
    store = PostgresCheckpointStore(pool, [], timeout_s=1.0)

    assert not await store.is_available()

    pool.fails = False

    assert await store.is_available()


@pytest.mark.faults
async def test_the_postgres_fault_switch_uses_the_fallback():
    store = PostgresCheckpointStore(FakePool(), [], timeout_s=1.0, enabled_faults=("postgres",))

    assert not await store.is_available()


@pytest.mark.faults
async def test_tables_are_created_after_a_late_start():
    # Postgres was down at startup, so setup must run when it first answers.
    setups = []

    async def record_setup():
        setups.append(1)

    pool = FakePool(fails=True)
    store = PostgresCheckpointStore(pool, [record_setup], timeout_s=1.0)
    await store.is_available()
    pool.fails = False

    await store.is_available()

    assert setups == [1]
