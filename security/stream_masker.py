"""Mask identity and card numbers, including while an answer is still streaming.

Rule 9: tokens reach the user as a live draft, before validation. A number can be split across
tokens, so masking each token alone would miss it. `StreamMasker` therefore holds back any text
at the end of its buffer that could still become a sensitive number.
"""

import re

# A card number: 13 to 19 digits together, or four-digit groups separated by spaces or dashes.
CARD_NUMBER_PATTERN = re.compile(r"\b(?:\d{13,19}|\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7})\b")
# A national identity number: 12 digits, or 9 digits followed by a letter.
NATIONAL_ID_PATTERN = re.compile(r"\b(?:\d{12}|\d{9}[VvXx])\b")
# Digits at the end of the buffer, which more tokens could extend into a sensitive number.
INCOMPLETE_NUMBER_PATTERN = re.compile(r"\d[\d -]*[VvXx]?$")

VISIBLE_CHARACTER_COUNT = 4


def _mask_card_number(number_match: re.Match[str]) -> str:
    """Return the masked form from the bank's masking standard: only the last four digits."""
    digits = re.sub(r"\D", "", number_match.group())
    return f"**** **** **** {digits[-VISIBLE_CHARACTER_COUNT:]}"


def _mask_national_id(number_match: re.Match[str]) -> str:
    """Return the number with all but its last four characters hidden."""
    national_id = number_match.group()
    hidden_length = len(national_id) - VISIBLE_CHARACTER_COUNT
    return "*" * hidden_length + national_id[hidden_length:]


def mask_sensitive_text(text: str) -> str:
    """Mask every complete sensitive number in the text.

    Card numbers go first, because the longer pattern must win over the identity pattern.
    """
    masked_text = CARD_NUMBER_PATTERN.sub(_mask_card_number, text)
    return NATIONAL_ID_PATTERN.sub(_mask_national_id, masked_text)


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
