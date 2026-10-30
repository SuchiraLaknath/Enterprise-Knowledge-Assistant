"""Mask identity and card numbers, including while an answer is still streaming (rule 9).

Tokens reach the user as a live draft, before validation. A number can be split across tokens,
so masking each token alone would miss it. `StreamMasker` therefore holds back any text at the
end of its buffer that could still become a sensitive number. This delays ordinary numbers such
as "step 3" by a token or two, which is a small price for never showing part of a card number.

A number is found as a run of digit groups, and judged by the lengths of those groups. Group
lengths separate a card number from ordinary figures such as dates, which a digit count cannot.
"""

import re

# Characters that can sit between digit groups, including the unicode spaces and dashes that a
# model may write, and the asterisks of markdown bold. Python's \s covers the unicode spaces.
SEPARATOR_CHARACTERS = r"\s\-./_*\u2010-\u2015\u200b"
SEPARATOR = f"[{SEPARATOR_CHARACTERS}]"

# Digit groups joined by short separators, with an optional letter, as in 901234567V.
NUMBER_RUN_PATTERN = re.compile(
    rf"(?<!\d)(?P<digits>\d+(?:{SEPARATOR}{{1,3}}\d+)*)"
    rf"(?P<letter>{SEPARATOR}?[VvXx](?![A-Za-z]))?"
)
# Digits at the end of the buffer, which more tokens could extend into a sensitive number.
INCOMPLETE_NUMBER_PATTERN = re.compile(rf"\d[\d{SEPARATOR_CHARACTERS}]*[VvXx]?$")
DIGIT_GROUP_PATTERN = re.compile(r"\d+")

VISIBLE_CHARACTER_COUNT = 4
# A national identity number has 12 digits, and a card number has 13 or more.
SHORTEST_SENSITIVE_DIGIT_GROUP = 12
SHORTEST_CARD_NUMBER = 13
# Group lengths of a spaced card or identity number, such as 4111 1111 1111 1111.
SENSITIVE_GROUP_SHAPES = ((4, 4, 4), (4, 6, 5), (4, 6, 4))
OLD_IDENTITY_NUMBER_LENGTH = 9
OLD_IDENTITY_NUMBER_SHAPE = (3, 3, 3)


def _has_sensitive_shape(group_lengths: list[int]) -> bool:
    """Return True if any neighbouring groups form a card or identity number."""
    if max(group_lengths) >= SHORTEST_SENSITIVE_DIGIT_GROUP:
        return True
    return any(
        tuple(group_lengths[start : start + len(shape)]) == shape
        for shape in SENSITIVE_GROUP_SHAPES
        for start in range(len(group_lengths))
    )


def _is_old_identity_number(group_lengths: list[int], letter: str) -> bool:
    """Return True for nine digits with a letter, such as 901234567V or 901 234 567 V."""
    if not letter:
        return False
    is_one_group = group_lengths[-1] == OLD_IDENTITY_NUMBER_LENGTH
    return is_one_group or tuple(group_lengths[-3:]) == OLD_IDENTITY_NUMBER_SHAPE


def _mask_number_run(number_match: re.Match[str]) -> str:
    """Mask a whole run if it holds a sensitive number, and leave it unchanged otherwise.

    The whole run is hidden, because a neighbouring ticket number must not shift which digits
    stay visible.
    """
    letter = number_match.group("letter") or ""
    digit_groups = DIGIT_GROUP_PATTERN.findall(number_match.group("digits"))
    group_lengths = [len(group) for group in digit_groups]
    digits = "".join(digit_groups)

    if _is_old_identity_number(group_lengths, letter):
        identity_number = digits + letter.strip()
        hidden_length = len(identity_number) - VISIBLE_CHARACTER_COUNT
        return "*" * hidden_length + identity_number[hidden_length:]

    if not _has_sensitive_shape(group_lengths):
        return number_match.group()

    last_digits = digits[-VISIBLE_CHARACTER_COUNT:]
    if len(digits) >= SHORTEST_CARD_NUMBER:
        return f"**** **** **** {last_digits}{letter}"
    return "*" * (len(digits) - VISIBLE_CHARACTER_COUNT) + last_digits + letter


def mask_sensitive_text(text: str) -> str:
    """Mask every complete sensitive number in the text."""
    return NUMBER_RUN_PATTERN.sub(_mask_number_run, text)


class StreamMasker:
    """Masks a stream of text fragments. Create a new one for each draft."""

    def __init__(self) -> None:
        """Start with nothing held back."""
        self._held_text = ""

    def feed(self, text_fragment: str) -> str:
        """Add a fragment, and return the text that is now safe to show."""
        self._held_text += text_fragment
        incomplete_number = INCOMPLETE_NUMBER_PATTERN.search(self._held_text)
        safe_length = incomplete_number.start() if incomplete_number else len(self._held_text)

        safe_text = self._held_text[:safe_length]
        self._held_text = self._held_text[safe_length:]
        return mask_sensitive_text(safe_text)

    def flush(self) -> str:
        """Return whatever is still held back, once the stream has ended."""
        remaining_text = self._held_text
        self._held_text = ""
        return mask_sensitive_text(remaining_text)
