from langchain_core.messages import HumanMessage

from api.tracing import redact_document_text
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
