"""What tool wrappers send to LangSmith, which is an external service.

Arguments can hold personal data or generated code, so traces show that a tool ran and how it
ended, but not what it was given.
"""

from typing import Any


def hide_tool_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Drop a tool's inputs from its trace."""
    return {}
