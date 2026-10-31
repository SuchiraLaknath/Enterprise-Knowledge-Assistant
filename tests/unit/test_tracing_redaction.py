from langchain_core.messages import HumanMessage

from agent.schemas import ExtractedFinding, ResearchReport, ResearchTheme
from api.tracing import anonymize_trace_payload, redact_document_text
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
        documents_in_scope=1,
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


def test_a_reply_cut_off_inside_a_claim_is_redacted():
    # Attack: a poisoned document asks to be quoted at length, so the reply hits the token limit.
    cut_off_reply = f'{{"findings": [{{"claim": "{SECRET_TEXT} and it goes on'

    assert SECRET_TEXT not in str(redact_document_text({"content": cut_off_reply}))


def test_a_theme_summary_is_hidden_in_every_form():
    # A summary paraphrases the claims, so hiding only the claims would achieve little.
    theme = {"name": "Expiry", "summary": SECRET_TEXT}
    raw_reply = f'{{"themes": [{{"name": "Expiry", "summary": "{SECRET_TEXT}"}}]}}'
    prompt = f'<theme name="Expiry" documents="2" summary="{SECRET_TEXT}">'

    assert SECRET_TEXT not in str(redact_document_text({"themes": [theme]}))
    assert SECRET_TEXT not in str(redact_document_text({"content": raw_reply}))
    assert SECRET_TEXT not in str(redact_document_text({"content": prompt}))
    assert "Expiry" in str(redact_document_text({"content": prompt}))


def test_an_error_keeps_only_its_type():
    # A parser's error quotes the whole reply, which can hold text from a document.
    error = f"OutputParserException('Failed to parse BatchAnalysis from {SECRET_TEXT}')\nTraceback"

    assert anonymize_trace_payload({"error": error}) == {"error": "OutputParserException"}


def test_an_ordinary_payload_is_redacted_by_the_anonymizer():
    payload = {"messages": [f'<document id="1" a="b">{SECRET_TEXT}</document>'], "error": None}

    assert SECRET_TEXT not in str(anonymize_trace_payload(payload))


def test_tool_result_bodies_are_hidden_from_prompts():
    prompt = f'<tool_result name="mcp.incident_records" status="ok">{SECRET_TEXT}</tool_result>'

    redacted = redact_document_text({"content": prompt})

    assert SECRET_TEXT not in str(redacted)
    assert "mcp.incident_records" in str(redacted)


def test_a_tool_result_in_state_loses_its_data_and_keeps_its_outcome():
    payload = {"tool_name": "python_analysis", "status": "ok", "data": SECRET_TEXT}

    assert redact_document_text(payload) == {"tool_name": "python_analysis", "status": "ok"}


def test_memory_is_hidden_from_prompts_and_state():
    prompt = f"Known: <memory>\n{SECRET_TEXT}\n</memory>"
    state = {"memory_context": SECRET_TEXT, "conversation_summary": SECRET_TEXT}

    assert SECRET_TEXT not in str(redact_document_text({"content": prompt}))
    assert SECRET_TEXT not in str(redact_document_text(state))


def test_tool_call_arguments_are_hidden():
    message = {"tool_calls": [{"name": "python_analysis", "args": {"code": SECRET_TEXT}}]}

    assert SECRET_TEXT not in str(redact_document_text(message))


def test_excerpts_are_hidden_as_they_are_raw_document_text():
    state = {"answer_is_excerpts": True, "final_answer": SECRET_TEXT}

    assert SECRET_TEXT not in str(redact_document_text(state))


def test_an_ordinary_answer_stays_visible():
    # The owner decided that traces should show answers, so evaluators can follow a turn.
    state = {"answer_is_excerpts": False, "final_answer": "A certificate expired [1]."}

    assert "A certificate expired" in str(redact_document_text(state))
