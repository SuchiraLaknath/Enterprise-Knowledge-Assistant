"""The deterministic half of the input guard: known attack patterns.

These run first, because they are instant, free and cannot be talked out of a decision. The
`input_guard` node adds an LLM judge for what patterns cannot recognise. If that judge is down,
these patterns decide alone, and tools are limited to reading.
"""

from security.policies import find_matching_category, get_security_policies


def find_input_attack(message: str) -> str | None:
    """Return the attack category of a message, or None if no pattern matches."""
    return find_matching_category(message, get_security_policies().input_patterns)
