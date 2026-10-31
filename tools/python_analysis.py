"""`python_analysis`: structured analysis on data this turn already retrieved.

The model writes short Python, and the one sandbox runs it (rule 7). It gets data-only helpers
and no retrieval, so it can only work on what this turn already found. `records` holds metadata
and tool results, never document text.
"""

import json
from collections import Counter
from typing import Any

from langsmith import traceable
from pydantic import Field

from security.sandbox import run_sandboxed
from tools.models import ToolArguments, ToolContext, ToolResult
from tools.tracing import hide_tool_inputs

TOOL_NAME = "python_analysis"
# The variable that the code must set. Everything else it creates is discarded.
RESULT_VARIABLE = "result"
NO_RESULT_MESSAGE = "The code must store its answer in a variable named result."


class PythonAnalysisArguments(ToolArguments):
    """The code to run, which the sandbox checks before anything executes.

    The description lists the syntax for the model, and must match `security/sandbox.py`.
    """

    code: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "Python that stores its answer in `result`. It can read `records`, a list of dicts, "
            "and call count_by(records, key), average(values) and percentage(part, whole). "
            "Allowed: assignment, for, if, comprehensions, + - / //. Not allowed: import, "
            "while, def, lambda, *, **, %."
        ),
    )


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count records by the value of one field, such as incidents per department."""
    return dict(Counter(str(record.get(key)) for record in records))


def average(values: list[float]) -> float:
    """Return the mean, or 0 for no values, so that empty data is not an error."""
    values_as_list = list(values)
    return sum(values_as_list) / len(values_as_list) if values_as_list else 0.0


def percentage(part: float, whole: float) -> float:
    """Return `part` as a percentage of `whole`. The sandbox has no `*`, so this is a helper."""
    return round(part / whole * 100, 2) if whole else 0.0


@traceable(name=TOOL_NAME, run_type="tool", process_inputs=hide_tool_inputs)
async def run_python_analysis(
    arguments: PythonAnalysisArguments, context: ToolContext
) -> ToolResult:
    """Run the code in the sandbox, and return only its `result`, cut to a safe length."""
    # Rule 7: data-only helpers, so this code has no search, no fetch and no sub-agents.
    data_only_names = {
        "records": context.analysis_records,
        "count_by": count_by,
        "average": average,
        "percentage": percentage,
    }
    variables = await run_sandboxed(arguments.code, data_only_names, context.sandbox_limits)
    if RESULT_VARIABLE not in variables:
        return ToolResult(tool_name=TOOL_NAME, status="error", message=NO_RESULT_MESSAGE)

    # A cap, because a large result would otherwise flood the prompt that reads it.
    result_text = json.dumps(variables[RESULT_VARIABLE], default=str)
    return ToolResult(
        tool_name=TOOL_NAME,
        status="ok",
        data=result_text[: context.tool_limits.max_result_characters],
    )
