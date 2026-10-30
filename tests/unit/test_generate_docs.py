from collections import Counter
from datetime import timedelta

import pytest

from ingestion.generate_docs import CORPUS_REFERENCE_DATE, build_corpus, write_corpus

ONE_YEAR_AGO = CORPUS_REFERENCE_DATE - timedelta(days=365)
MINIMUM_DOCUMENT_WORDS = 300


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


def test_corpus_is_about_eighty_documents(corpus):
    assert 75 <= len(corpus) <= 90


def test_generation_is_reproducible(corpus):
    assert [document.to_markdown() for document in build_corpus()] == [
        document.to_markdown() for document in corpus
    ]


def test_document_ids_are_unique(corpus):
    document_ids = [document.doc_id for document in corpus]

    assert len(document_ids) == len(set(document_ids))


def test_all_departments_document_types_and_access_levels_are_present(corpus):
    assert {document.department for document in corpus} == {
        "payments",
        "platform",
        "security",
        "hr-policy",
        "product",
    }
    assert {document.document_type for document in corpus} == {
        "policy",
        "architecture",
        "runbook",
        "incident",
        "product_spec",
        "meeting_notes",
    }
    assert {document.access_level for document in corpus} == {
        "public",
        "internal",
        "confidential",
        "restricted",
    }


def test_at_least_24_payment_incidents_fall_in_the_last_12_months(corpus):
    recent_payment_incidents = [
        document
        for document in corpus
        if document.document_type == "incident"
        and document.department == "payments"
        and document.created_date >= ONE_YEAR_AGO
    ]

    assert len(recent_payment_incidents) >= 24


def test_older_payment_incidents_exist_for_date_filtering(corpus):
    older_incidents = [
        document
        for document in corpus
        if document.document_type == "incident" and document.created_date < ONE_YEAR_AGO
    ]

    assert len(older_incidents) == 3


def test_each_seeded_root_cause_recurs_across_payment_incidents(corpus):
    error_code_counts = Counter(
        error_code
        for document in corpus
        if document.document_type == "incident"
        for error_code in ("ERR-PAY-4091", "ERR-PAY-5032", "ERR-PAY-5041")
        if error_code in document.body
    )

    assert error_code_counts["ERR-PAY-4091"] >= 8
    assert error_code_counts["ERR-PAY-5032"] >= 8
    assert error_code_counts["ERR-PAY-5041"] >= 6


def test_incidents_exist_at_each_elevated_access_level(corpus):
    incident_access_levels = {
        document.access_level for document in corpus if document.document_type == "incident"
    }

    assert {"confidential", "restricted"} <= incident_access_levels


def test_exactly_one_document_carries_the_poisoned_instruction(corpus):
    poisoned_documents = [
        document for document in corpus if "Ignore all previous instructions" in document.body
    ]

    assert [document.doc_id for document in poisoned_documents] == ["RB-006"]


def test_every_document_has_enough_text_to_search(corpus):
    short_documents = [
        document.doc_id
        for document in corpus
        if len(document.body.split()) < MINIMUM_DOCUMENT_WORDS
    ]

    assert short_documents == []


def test_markdown_starts_with_front_matter(corpus):
    assert corpus[0].to_markdown().startswith("---\ndoc_id:")


def test_writing_removes_files_from_an_earlier_run(corpus, tmp_path):
    # A stale file from a moved document would return at its old access level.
    stale_file = tmp_path / "old-department" / "MOVED.md"
    stale_file.parent.mkdir()
    stale_file.write_text("stale")

    write_corpus(corpus[:1], tmp_path)

    assert not stale_file.exists()
    assert len(list(tmp_path.rglob("*.md"))) == 1
