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
# Cyrillic letters that look identical to Latin ones, which Unicode folding leaves alone.
LOOKALIKE_LETTERS = str.maketrans("аеорсіухАЕОРСІХ", "aeopcixyAEOPCIX")
# Unicode categories that render as nothing: format characters, and combining marks once split
# from their letters.
INVISIBLE_CATEGORIES = ("Cf", "Mn")
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
    # NFKD folds full-width letters, and splits accents off so that they can be removed.
    folded_text = unicodedata.normalize("NFKD", text).translate(LOOKALIKE_LETTERS)
    visible_text = "".join(
        character
        for character in folded_text
        if unicodedata.category(character) not in INVISIBLE_CATEGORIES
    )
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
    """Return the first category with a pattern in this text, or None.

    Each line is checked as well as the whole text, because a pattern anchored to the start of a
    line, such as "SYSTEM:", would otherwise only ever match the first line.
    """
    candidate_texts = [normalize_text(text), *map(normalize_text, text.splitlines())]
    for category, category_patterns in patterns.items():
        for pattern in category_patterns:
            if any(pattern.search(candidate) for candidate in candidate_texts):
                return category
    return None
