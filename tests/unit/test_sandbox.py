import ast
import asyncio
import sys
import time

import pytest

from config.settings import SandboxLimits
from security import sandbox
from security.sandbox import SandboxTimeout, SandboxViolation, run_sandboxed, validate_code

LIMITS = SandboxLimits(cpu_seconds=0.05, max_range_items=10_000)
NO_HELPERS = frozenset()


@pytest.mark.parametrize(
    "attack",
    [
        # Reaching the operating system directly.
        "import os",
        "from os import system",
        "__import__('os')",
        "open('/etc/passwd')",
        "eval('1')",
        "exec('1')",
        "compile('1', 'x', 'exec')",
        "globals()",
        "locals()",
        "vars()",
        "getattr(str, 'join')",
        "breakpoint()",
        "print('leak')",
        # Walking from a plain object to the interpreter.
        "().__class__",
        "''.__class__.__mro__",
        "[].__class__.__base__.__subclasses__()",
        "x = 1\nx.__class__",
        "__builtins__",
        "__name__",
        # Walking there through a string, which a dunder ban alone would miss.
        "'{0.__class__}'.format(1)",
        "'{0.__class__}'.format_map({})",
        "f'{().__class__}'",
        # Generator and frame internals have no dunders at all.
        "(x for x in []).gi_frame",
        "(x for x in []).gi_code",
        # Syntax that could hide logic or swallow the stop signal.
        "while True:\n    pass",
        "def run():\n    pass",
        "class Plan:\n    pass",
        "(lambda: 1)()",
        "try:\n    pass\nexcept Exception:\n    pass",
        "with open('x') as f:\n    pass",
        "async def run():\n    pass",
        "(y := 1)",
        "del len",
        "global x",
        "raise ValueError()",
        "yield 1",
        "assert False",
        # Huge objects built in C, where the line trace cannot stop them.
        "2 ** 100000000",
        "[0] * 10000000000",
        "'%999999999d' % 1",
        "f'{1:>999999999}'",
        # Names that were never provided or assigned.
        "unknown_helper()",
    ],
)
def test_attacks_are_refused_before_running(attack):
    with pytest.raises(SandboxViolation):
        validate_code(attack, NO_HELPERS)


def test_broken_python_is_refused():
    with pytest.raises(SandboxViolation):
        validate_code("for", NO_HELPERS)


def test_plain_data_code_is_accepted():
    code = """
titles = [title.lower() for title in ["A", "B"] if title != "B"]
counts = {}
for title in titles:
    counts[title] = counts.get(title, 0) + 1
total = sum(counts.values())
"""

    validate_code(code, NO_HELPERS)


async def test_code_runs_and_returns_its_variables():
    variables = await run_sandboxed("total = sum([1, 2, 3])", {}, LIMITS)

    assert variables == {"total": 6}


async def test_code_can_call_only_the_helpers_it_was_given():
    variables = await run_sandboxed("answer = add_one(1)", {"add_one": lambda x: x + 1}, LIMITS)

    assert variables == {"answer": 2}
    with pytest.raises(SandboxViolation):
        await run_sandboxed("answer = other_helper(1)", {"add_one": lambda x: x + 1}, LIMITS)


async def test_a_refused_program_has_no_effect_at_all():
    # The attack sits on the last line, so nothing before it may run either.
    calls = []

    with pytest.raises(SandboxViolation):
        await run_sandboxed("record(1)\nimport os", {"record": calls.append}, LIMITS)

    assert calls == []


async def test_real_builtins_are_missing_even_if_validation_is_bypassed(monkeypatch):
    # Attack: suppose a validator bug lets `open` through. The second layer must still hold.
    monkeypatch.setattr(sandbox, "validate_code", lambda source, names: ast.parse(source))

    with pytest.raises(NameError):
        await run_sandboxed("open('/etc/passwd')", {}, LIMITS)


async def test_a_cpu_loop_is_stopped():
    code = """
total = 0
for first in range(10000):
    for second in range(10000):
        total = total + 1
"""

    with pytest.raises(SandboxTimeout):
        await run_sandboxed(code, {}, LIMITS)


async def test_a_loop_on_one_line_is_stopped():
    # Attack: a single line, in case the trace only fires when the line number changes.
    code = "values = sum(1 for first in range(10000) for second in range(10000))"

    with pytest.raises(SandboxTimeout):
        await run_sandboxed(code, {}, LIMITS)


async def test_a_list_growing_while_it_is_read_is_stopped():
    # Attack: an endless `for` loop, which needs no `while`.
    code = "items = [1]\nfor item in items:\n    items.append(item)"

    with pytest.raises(SandboxTimeout):
        await run_sandboxed(code, {}, LIMITS)


async def test_a_huge_range_is_refused():
    # Attack: `sorted(range(n))` runs only in C, so the line trace could not stop it.
    with pytest.raises(SandboxViolation):
        await run_sandboxed("values = sorted(range(10000000000))", {}, LIMITS)


async def test_waiting_on_a_helper_uses_no_cpu_budget():
    # The helper waits ten times longer than the whole budget, as a slow search would.
    def wait_like_a_slow_search() -> str:
        time.sleep(0.5)
        return "done"

    variables = await run_sandboxed(
        "result = wait_like_a_slow_search()",
        {"wait_like_a_slow_search": wait_like_a_slow_search},
        LIMITS,
    )

    assert variables == {"result": "done"}


async def test_a_helper_can_use_the_event_loop_while_code_runs():
    # Helpers are synchronous, but searches are async, so they hop back to the loop.
    event_loop = asyncio.get_running_loop()

    async def fetch_value() -> int:
        await asyncio.sleep(0)
        return 7

    def call_async_helper() -> int:
        return asyncio.run_coroutine_threadsafe(fetch_value(), event_loop).result(timeout=5)

    variables = await run_sandboxed(
        "value = call_async_helper()", {"call_async_helper": call_async_helper}, LIMITS
    )

    assert variables == {"value": 7}


def test_the_trace_is_removed_after_a_run():
    # A leftover trace would keep stopping unrelated code that later uses this thread.
    compiled_code = compile("total = 1", sandbox.SANDBOX_FILENAME, "exec")

    sandbox._execute_with_cpu_limit(compiled_code, {"__builtins__": {}}, LIMITS.cpu_seconds)

    assert sys.gettrace() is None
