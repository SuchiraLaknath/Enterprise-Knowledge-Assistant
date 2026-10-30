import pytest

from config.settings import ChunkingLimits, get_limits
from ingestion.chunk import chunk_document, parse_markdown_document
from ingestion.generate_docs import build_corpus

TEST_LIMITS = ChunkingLimits(
    min_tokens=13,
    max_tokens=52,
    overlap_fraction=0.15,
    tokens_per_word=1.3,
)


def make_document(body: str):
    return parse_markdown_document(
        "---\n"
        "doc_id: TEST-1\n"
        "title: Test document\n"
        "department: payments\n"
        "document_type: runbook\n"
        "access_level: internal\n"
        "created_date: '2026-01-15'\n"
        "source_uri: docs://payments/TEST-1\n"
        f"---\n\n{body}\n"
    )


def repeated_words(word: str, count: int) -> str:
    return " ".join([word] * count)


def test_parse_reads_front_matter_and_body():
    document = make_document("# Title\n\nSome text")

    assert document.doc_id == "TEST-1"
    assert document.access_level == "internal"
    assert document.body == "# Title\n\nSome text"


def test_parse_rejects_a_document_without_front_matter():
    with pytest.raises(ValueError, match="front-matter"):
        parse_markdown_document("# Just a heading")


def test_chunk_ids_follow_the_documented_format():
    body = f"## One\n\n{repeated_words('alpha', 30)}\n\n## Two\n\n{repeated_words('beta', 30)}"

    chunks = chunk_document(make_document(body), TEST_LIMITS)

    assert [chunk.chunk_id for chunk in chunks] == ["TEST-1#c0", "TEST-1#c1"]


def test_section_keeps_the_heading_path():
    body = f"## Parent\n\n### Child\n\n{repeated_words('alpha', 20)}"

    chunks = chunk_document(make_document(body), TEST_LIMITS)

    assert chunks[0].section == "Parent > Child"


def test_small_neighbouring_sections_merge_and_keep_both_names():
    body = "## First\n\nfew words here\n\n## Second\n\nmore words here"

    chunks = chunk_document(make_document(body), TEST_LIMITS)

    assert len(chunks) == 1
    assert chunks[0].section == "First; Second"


def test_a_long_section_is_split_with_every_word_kept():
    long_text = " ".join(f"word{number}" for number in range(100))

    chunks = chunk_document(make_document(f"## Long\n\n{long_text}"), TEST_LIMITS)

    combined_text = " ".join(chunk.text for chunk in chunks)
    assert len(chunks) > 1
    assert all(f"word{number}" in combined_text.split() for number in range(100))


def test_next_chunk_repeats_the_end_of_the_previous_chunk():
    long_text = " ".join(f"word{number}" for number in range(100))

    first_chunk, second_chunk, *_ = chunk_document(
        make_document(f"## Long\n\n{long_text}"), TEST_LIMITS
    )

    last_word_of_first_chunk = first_chunk.text.split()[-1]
    assert last_word_of_first_chunk in second_chunk.text.split()


def test_metadata_is_complete_and_the_timestamp_is_numeric():
    chunk = chunk_document(make_document("## One\n\nsome words"), TEST_LIMITS)[0]

    assert chunk.department == "payments"
    assert chunk.document_type == "runbook"
    assert chunk.created_date == "2026-01-15"
    assert chunk.created_ts == 1768435200
    assert chunk.source_uri == "docs://payments/TEST-1"


def test_embedding_text_includes_title_and_section():
    chunk = chunk_document(make_document("## One\n\nsome words"), TEST_LIMITS)[0]

    assert chunk.text_for_embedding().startswith("Test document > One\n")


def test_real_corpus_chunks_stay_within_the_token_limit():
    limits = get_limits().chunking
    all_chunks = [
        chunk
        for generated_document in build_corpus()
        for chunk in chunk_document(
            parse_markdown_document(generated_document.to_markdown()), limits
        )
    ]

    largest_chunk_tokens = max(
        len(chunk.text.split()) * limits.tokens_per_word for chunk in all_chunks
    )
    assert largest_chunk_tokens <= limits.max_tokens


def test_most_real_documents_split_into_several_chunks():
    limits = get_limits().chunking
    chunk_counts = [
        len(chunk_document(parse_markdown_document(document.to_markdown()), limits))
        for document in build_corpus()
    ]

    documents_with_several_chunks = [count for count in chunk_counts if count >= 2]
    assert len(documents_with_several_chunks) > len(chunk_counts) * 0.8
