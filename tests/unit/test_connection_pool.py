from unittest.mock import patch

from api.main import create_connection_pool


def test_pool_tests_each_connection_before_lending_it_out():
    # Simulates Postgres restarting while the API runs: without this check, every idle
    # connection is dead, and each one fails a chat turn before the pool replaces it.
    with patch("api.main.AsyncConnectionPool") as pool_class:
        create_connection_pool("postgresql://unused", 2.0)

    assert pool_class.call_args.kwargs["check"] is pool_class.check_connection
