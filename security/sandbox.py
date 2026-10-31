"""The one sandbox for generated Python, shared by RLM plans and `python_analysis` (rule 7).

Generated code is untrusted, because a prompt injection could shape what a model writes. Two
layers contain it. Before running, `validate_code` accepts only an allow-list of syntax, names
and attributes, so that imports, dunder access and `while` never start. While running, a line
trace stops the code after a budget of CPU time, in its own thread, so that time spent waiting
on a helper does not count.

Known limit: the trace runs between Python lines, so one long operation in C cannot be stopped
partway. `Mult`, `Pow` and `Mod` are refused and `range` is capped for that reason. See
`docs/adr/` for what remains.
"""

import ast
import asyncio
import contextvars
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from types import CodeType, FrameType
from typing import Any

from config.settings import SandboxLimits

# Compiled code carries this name, which is how the line trace recognises it.
SANDBOX_FILENAME = "<sandboxed-code>"
THREAD_NAME_PREFIX = "sandbox"

# Everything not listed is refused, including imports, `while`, functions, classes and `try`.
ALLOWED_SYNTAX_NODES: tuple[type[ast.AST], ...] = (
    # Statements.
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.For,
    ast.If,
    ast.Break,
    ast.Continue,
    ast.Pass,
    # Values and containers.
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.JoinedStr,
    ast.FormattedValue,
    # Calls and attributes. Attribute names are checked separately.
    ast.Call,
    ast.keyword,
    ast.Attribute,
    # Comprehensions, which a plan uses to filter a list.
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    # Conditions.
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.IfExp,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    # Arithmetic. `Mult`, `Pow` and `Mod` are left out, because `[0] * 10000000000` and
    # `"%999999999d" % 1` would build a huge object in C, where the line trace cannot stop them.
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Div,
    ast.FloorDiv,
)

# Rule 7: a dunder ban alone would miss `"{0.__class__}".format(value)`, which reaches the same
# objects through a string. Only plain data methods are allowed, and `format` is not one.
ALLOWED_ATTRIBUTES = frozenset(
    {
        "add",
        "append",
        "count",
        "endswith",
        "extend",
        "get",
        "items",
        "join",
        "keys",
        "lower",
        "split",
        "startswith",
        "strip",
        "title",
        "upper",
        "values",
    }
)

# Plain data functions with no route to files, the network or the interpreter. `range` is added
# separately, with a cap.
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
BUILTIN_RANGE_NAME = "range"


class SandboxViolation(Exception):
    """The code uses something the sandbox refuses, so it never started or was stopped."""


class SandboxTimeout(Exception):
    """The code used up its CPU budget."""


def _is_dunder(name: str) -> bool:
    """Return whether a name is a Python internal, such as `__class__`."""
    return name.startswith("__")


def _find_assigned_names(syntax_tree: ast.Module) -> set[str]:
    """Return the variables the code creates, which it may then read."""
    assigned_names = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned_names.add(node.id)
    return assigned_names


def _check_node(node: ast.AST, readable_names: set[str]) -> None:
    """Raise if one syntax node breaks the allow-list (rule 7)."""
    if not isinstance(node, ALLOWED_SYNTAX_NODES):
        raise SandboxViolation(f"{type(node).__name__} is not allowed.")

    if isinstance(node, ast.Name):
        if _is_dunder(node.id):
            raise SandboxViolation(f"The name {node.id} is not allowed.")
        if isinstance(node.ctx, ast.Load) and node.id not in readable_names:
            raise SandboxViolation(f"The name {node.id} is not available.")

    if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_ATTRIBUTES:
        raise SandboxViolation(f"The attribute {node.attr} is not allowed.")

    # A width such as f"{value:>999999999}" would build a huge string in C.
    if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
        raise SandboxViolation("Format specifications are not allowed.")


def validate_code(source: str, allowed_names: frozenset[str]) -> ast.Module:
    """Return the parsed code, or raise `SandboxViolation` if any part of it is refused.

    The check runs before anything executes, so refused code has no effect at all.
    """
    try:
        syntax_tree = ast.parse(source, filename=SANDBOX_FILENAME)
    except SyntaxError as error:
        raise SandboxViolation(f"The code is not valid Python: {error.msg}") from error

    readable_names = {*allowed_names, *SAFE_BUILTINS, BUILTIN_RANGE_NAME}
    readable_names |= _find_assigned_names(syntax_tree)
    for node in ast.walk(syntax_tree):
        _check_node(node, readable_names)
    return syntax_tree


def _build_range_with_a_cap(max_range_items: int) -> Callable[..., range]:
    """Return a `range` that refuses huge sizes, as `sorted(range(n))` would run only in C."""

    def capped_range(*range_arguments: int) -> range:
        requested_range = range(*range_arguments)
        if len(requested_range) > max_range_items:
            raise SandboxViolation(f"A range may hold at most {max_range_items} items.")
        return requested_range

    return capped_range


def _build_cpu_limit_trace(cpu_seconds: float) -> Callable[..., Any]:
    """Return a trace function that stops sandboxed code after `cpu_seconds` of CPU time.

    `time.thread_time` counts only this thread's work, so a helper that waits on the network
    uses none of the budget.
    """
    cpu_deadline = time.thread_time() + cpu_seconds

    def check_cpu_on_each_line(frame: FrameType, event: str, argument: Any) -> Callable[..., Any]:
        if event == "line" and time.thread_time() > cpu_deadline:
            raise SandboxTimeout(f"The code used more than {cpu_seconds} s of CPU time.")
        return check_cpu_on_each_line

    def trace_only_sandboxed_frames(
        frame: FrameType, event: str, argument: Any
    ) -> Callable[..., Any] | None:
        # Helpers are trusted code, and tracing them would only slow them down.
        if frame.f_code.co_filename == SANDBOX_FILENAME:
            return check_cpu_on_each_line
        return None

    return trace_only_sandboxed_frames


def _execute_with_cpu_limit(
    compiled_code: CodeType, namespace: dict[str, Any], cpu_seconds: float
) -> None:
    """Run the code in this thread, always removing the trace afterwards."""
    sys.settrace(_build_cpu_limit_trace(cpu_seconds))
    try:
        exec(compiled_code, namespace)
    finally:
        sys.settrace(None)


async def run_sandboxed(
    source: str, helpers: Mapping[str, Any], limits: SandboxLimits
) -> dict[str, Any]:
    """Validate and run generated code, and return the variables it created.

    Each run gets a dedicated thread, because the trace and `time.thread_time` both apply to
    one thread. The thread runs in a copy of this context, so that helpers keep the stream
    writer and the trace of the current turn.
    """
    # Step 1: rule 7, refuse the code before any of it runs.
    syntax_tree = validate_code(source, frozenset(helpers))
    compiled_code = compile(syntax_tree, SANDBOX_FILENAME, "exec")

    # Step 2: the code sees only the safe builtins and the helpers it was given.
    safe_builtins = {
        **SAFE_BUILTINS,
        BUILTIN_RANGE_NAME: _build_range_with_a_cap(limits.max_range_items),
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, **helpers}

    # Step 3: run it off the event loop, so that helpers can use the loop while it runs.
    event_loop = asyncio.get_running_loop()
    thread_context = contextvars.copy_context()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=THREAD_NAME_PREFIX)
    try:
        await event_loop.run_in_executor(
            executor,
            thread_context.run,
            _execute_with_cpu_limit,
            compiled_code,
            namespace,
            limits.cpu_seconds,
        )
    finally:
        # Not waiting means that a cancelled turn cannot block the event loop on this thread.
        executor.shutdown(wait=False)

    return {
        name: value
        for name, value in namespace.items()
        if name != "__builtins__" and name not in helpers
    }
