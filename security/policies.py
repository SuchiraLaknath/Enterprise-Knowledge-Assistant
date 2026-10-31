"""Load `security/policies.yaml`, and prepare text so that one pattern covers many spellings.

Attackers hide words with invisible characters, Unicode look-alikes and odd spacing. Normalising
first means that each pattern needs only one plain spelling, which keeps the policy readable.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

POLICIES_PATH = Path(__file__).parent / "policies.yaml"
# Characters that render as nothing, which split a word without changing how it looks: zero-width
# space, non-joiner and joiner, word joiner, byte-order mark and soft hyphen.
INVISIBLE_CHARACTERS = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
WHITESPACE_RUN = re.compile(r"\s+")

CompiledPatterns = dict[str, list[re.Pattern[str]]]


@dataclass(frozen=True)
class SecurityPolicies:
    """The compiled patterns of each guard, by category."""

    input_patterns: CompiledPatterns
    content_patterns: CompiledPatterns
    brand_patterns: CompiledPatterns


def normalize_text(text: str) -> str:
    """Return text in the one form that the patterns are written for."""
    folded_text = unicodedata.normalize("NFKC", text)
    visible_text = INVISIBLE_CHARACTERS.sub("", folded_text)
    return WHITESPACE_RUN.sub(" ", visible_text).strip().lower()


def _compile_section(section: dict[str, list[str]]) -> CompiledPatterns:
    """Compile one section of the policy file."""
    return {
        category: [re.compile(pattern) for pattern in patterns]
        for category, patterns in section.items()
    }


@lru_cache
def get_security_policies() -> SecurityPolicies:
    """Return the policies, read once per process."""
    with POLICIES_PATH.open(encoding="utf-8") as policies_file:
        raw_policies = yaml.safe_load(policies_file)
    return SecurityPolicies(
        input_patterns=_compile_section(raw_policies["input"]),
        content_patterns=_compile_section(raw_policies["content"]),
        brand_patterns=_compile_section(raw_policies["brand"]),
    )


def find_matching_category(text: str, patterns: CompiledPatterns) -> str | None:
    """Return the first category with a pattern in this text, or None."""
    normalized_text = normalize_text(text)
    for category, category_patterns in patterns.items():
        for pattern in category_patterns:
            if pattern.search(normalized_text):
                return category
    return None
