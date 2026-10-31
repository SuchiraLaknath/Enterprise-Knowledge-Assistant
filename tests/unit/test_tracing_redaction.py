from langchain_core.messages import HumanMessage

from agent.schemas import ExtractedFinding, ResearchReport, ResearchTheme
from api.tracing import redact_document_text
from tests.fakes.fake_research import make_finding
from tests.fakes.fake_search import make_chunk

SECRET_TEXT = "restricted document body"


def test_a_document_body_in_a_prompt_is_hidden():
    prompt = (
        f'Read this.\n<document id="1" title="T" section="S">\n{SECRET_TEXT}\n</document>\nDone.'
    )

    redacted = redact_document_text({"messages": [[{"content": prompt}]]})

    assert SECRET_TEXT not in str(redacted)
    assert '<document id="1" title="T" section="S">' in str(redacted)
    assert "Read this." in str(redacted)


def test_several_documents_are_all_hidden():
    prompt = (
        '<document id="1" a="b">one secret</document> and '
        '<document id="2" a="b">two secret</document>'
    )

    assert "secret" not in redact_document_text(prompt)


def test_a_chunk_object_in_state_loses_its_text_and_keeps_its_id():
    redacted = redact_document_text({"evidence": [make_chunk("DOC#c0", text=SECRET_TEXT)]})

    assert SECRET_TEXT not in str(redacted)
    assert "DOC#c0" in str(redacted)


def test_a_chunk_already_turned_into_a_dict_loses_its_text():
    redacted = redact_document_text({"chunk_id": "DOC#c0", "text": SECRET_TEXT, "score": 0.5})

    assert redacted == {"chunk_id": "DOC#c0", "score": 0.5}


def test_message_objects_are_redacted():
    message = HumanMessage(f'<document id="1" a="b">{SECRET_TEXT}</document>')

    assert SECRET_TEXT not in str(redact_document_text({"messages": [message]}))


def test_ordinary_values_pass_through():
    payload = {"question": "Why?", "count": 3, "ok": True, "nothing": None}

    assert redact_document_text(payload) == payload


def test_a_finding_in_state_loses_its_claim_and_keeps_its_evidence():
    # A live trace showed findings reaching LangSmith, and a claim often quotes its document.
    finding = make_finding("F1", "INC-1").model_copy(update={"claim": SECRET_TEXT})
    theme = ResearchTheme(
        name="Expiry",
        summary="Expired.",
        document_count=1,
        first_date="2026-01-15",
        last_date="2026-01-15",
        findings=[finding],
    )
    report = ResearchReport(
        themes=[theme],
        documents_analysed=1,
        chunks_read=1,
        chunks_available=1,
        depth_reached=1,
        sub_agent_calls=1,
    )

    redacted = str(redact_document_text({"findings": [finding], "research_report": report}))

    assert SECRET_TEXT not in redacted
    assert "INC-1#c0" in redacted
    assert "certificate expiry" in redacted


def test_a_structured_sub_agent_reply_is_redacted():
    reply = ExtractedFinding(claim=SECRET_TEXT, theme_label="expiry", evidence_markers=[1])

    assert SECRET_TEXT not in str(redact_document_text({"findings": [reply]}))


def test_a_claim_inside_a_raw_model_reply_is_redacted():
    # Before parsing, a sub-agent's reply is JSON held in a string, so no dict key can match.
    raw_reply = (
        f'{{"findings": [{{"claim": "{SECRET_TEXT} said \\"quoted\\"", "theme_label": "x"}}]}}'
    )

    redacted = redact_document_text({"content": raw_reply})

    assert SECRET_TEXT not in str(redacted)
    assert "quoted" not in str(redacted)
    assert "theme_label" in str(redacted)
