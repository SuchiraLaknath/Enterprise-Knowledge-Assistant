import pytest

from security.stream_masker import StreamMasker, mask_sensitive_text


def stream(fragments: list[str]) -> str:
    masker = StreamMasker()
    shown_text = "".join(masker.feed(fragment) for fragment in fragments)
    return shown_text + masker.flush()


@pytest.mark.parametrize(
    ("raw_text", "masked_text"),
    [
        ("Card 4111111111111111 used", "Card **** **** **** 1111 used"),
        ("Card 4111 1111 1111 1111 used", "Card **** **** **** 1111 used"),
        ("Card 4111-1111-1111-1111 used", "Card **** **** **** 1111 used"),
        ("NIC 199012345678 used", "NIC ********5678 used"),
        ("NIC 901234567V used", "NIC ******567V used"),
    ],
)
def test_sensitive_numbers_are_masked(raw_text, masked_text):
    assert mask_sensitive_text(raw_text) == masked_text


@pytest.mark.parametrize(
    "harmless_text",
    [
        "The outage on 2026-10-30 lasted 240 minutes.",
        "Between 2026-01-15 2026-02-03 there were 42000 failures.",
        "Error ERR-PAY-5041 was logged.",
        "Call 0112345678 for help.",
    ],
)
def test_ordinary_numbers_are_left_alone(harmless_text):
    assert mask_sensitive_text(harmless_text) == harmless_text


def test_a_card_number_split_across_tokens_never_appears():
    # Rule 9: each fragment looks harmless alone, so masking per token would leak the number.
    fragments = ["Your card ", "4111", " 1111 ", "1111", " 1111", " is valid"]

    shown_text = stream(fragments)

    assert shown_text == "Your card **** **** **** 1111 is valid"


def test_no_partial_digits_escape_before_the_number_is_complete():
    masker = StreamMasker()

    shown_while_streaming = masker.feed("NIC 1990") + masker.feed("1234")

    assert "1990" not in shown_while_streaming


def test_a_national_id_split_across_tokens_is_masked():
    assert stream(["NIC 19901", "2345", "678 is on file"]) == "NIC ********5678 is on file"


def test_a_national_id_with_a_letter_split_from_its_digits_is_masked():
    assert stream(["NIC 901234567", "V today"]) == "NIC ******567V today"


def test_a_number_at_the_end_of_the_stream_is_masked_on_flush():
    assert stream(["Card 4111111111111111"]) == "Card **** **** **** 1111"


def test_plain_text_passes_through_unchanged():
    assert stream(["Certificate ", "expiry caused ", "the outage."]) == (
        "Certificate expiry caused the outage."
    )
