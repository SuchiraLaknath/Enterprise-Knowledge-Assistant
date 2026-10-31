from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from streamlit.testing.v1 import AppTest

STREAMLIT_APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"
RENDER_TIMEOUT_SECONDS = 30


def render_page(signed_in):
    app = AppTest.from_file(STREAMLIT_APP_PATH, default_timeout=RENDER_TIMEOUT_SECONDS)
    if signed_in:
        app.session_state["access_token"] = "unused-token"
        app.session_state["username"] = "vimal"
        app.session_state["role"] = "viewer"
    return app.run()


def test_the_message_box_stays_outside_the_columns():
    # Inside a column, Streamlit places the message box in the page with the messages, so it
    # moves as the chat grows instead of staying at the bottom of the window.
    app = render_page(signed_in=True)

    assert not app.exception
    assert len(app.chat_input) == 1
    for column in app.columns:
        assert len(column.chat_input) == 0


def test_the_message_box_is_hidden_until_sign_in():
    app = render_page(signed_in=False)

    assert not app.exception
    assert len(app.chat_input) == 0
    assert [notice.value for notice in app.info] == ["Sign in from the sidebar to start."]


def make_streamed_response(status_code=200, event_lines=()):
    """Stand in for `httpx.stream`, with the lines an API reply would hold."""
    response = MagicMock(status_code=status_code, headers={})
    response.iter_lines.return_value = list(event_lines)
    stream = MagicMock()
    stream.__enter__.return_value = response
    return stream


APPROVAL_EVENT_LINE = (
    'data: {"type": "approval", "tool_call_id": "call-1", "tool_name": '
    '"admin.update_incident_status", "arguments": {"status": "resolved"}}'
)


def test_sign_in_explains_when_the_api_is_down():
    # Simulates the API being stopped: an uncaught error would fill the page with a stack trace.
    app = render_page(signed_in=False)

    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        app.sidebar.button[0].click().run()

    assert not app.exception
    assert [notice.value for notice in app.error] == [
        "The assistant could not be reached. Please try again."
    ]


def test_a_turn_that_pauses_for_approval_leaves_no_empty_bubble():
    # The turn ends with no answer, so its bubble would otherwise sit empty above the approval box.
    app = render_page(signed_in=True)

    with patch(
        "httpx.stream", return_value=make_streamed_response(event_lines=[APPROVAL_EVENT_LINE])
    ):
        app.chat_input[0].set_value("Close the incident").run()

    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    assert [notice.value for notice in app.warning] == [
        "This action changes data, and needs your approval. Nothing has changed yet."
    ]


def test_a_turn_that_ends_with_no_answer_says_so():
    # Simulates a stream that closes early: a silent empty bubble would look like a frozen app.
    app = render_page(signed_in=True)

    with patch("httpx.stream", return_value=make_streamed_response(event_lines=[])):
        app.chat_input[0].set_value("Hello").run()

    assert not app.exception
    assert [notice.value for notice in app.error] == ["No answer arrived. Please try again."]


def test_the_message_box_is_disabled_while_an_approval_waits():
    # A new message during a pause would leave the paused tool call undecided.
    app = render_page(signed_in=True)

    with patch(
        "httpx.stream", return_value=make_streamed_response(event_lines=[APPROVAL_EVENT_LINE])
    ):
        app.chat_input[0].set_value("Close the incident").run()

    assert app.chat_input[0].disabled is True


def test_an_expired_session_signs_the_user_out():
    # Simulates a token that ran out mid-demo: the sidebar must not go on saying "Signed in".
    app = render_page(signed_in=True)

    with patch("httpx.stream", return_value=make_streamed_response(status_code=401)):
        app.chat_input[0].set_value("Hello").run()

    assert not app.exception
    assert "access_token" not in app.session_state
    assert [notice.value for notice in app.sidebar.warning] == [
        "Your session has expired. Please sign in again."
    ]


def test_the_user_list_shows_each_type_of_user():
    app = render_page(signed_in=False)

    assert list(app.sidebar.selectbox[0].options) == [
        "vimal - viewer",
        "nadee - analyst",
        "admin - admin",
    ]


def test_the_sidebar_shows_the_signed_in_user_with_their_type():
    app = render_page(signed_in=True)

    assert any("vimal - viewer" in block.value for block in app.sidebar.markdown)


def test_an_empty_chat_offers_questions_that_send_when_clicked():
    app = render_page(signed_in=True)
    answer_line = 'data: {"type": "final", "answer": "Five days.", "citations": [], "degraded": []}'

    starter_button = next(button for button in app.button if button.key.startswith("starter_"))
    with patch(
        "httpx.stream", return_value=make_streamed_response(event_lines=[answer_line])
    ) as stream:
        starter_button.click().run()

    assert not app.exception
    assert (
        stream.call_args.kwargs["json"]["message"]
        == "How many days of annual leave can I carry over?"
    )
    # The chat is no longer empty, so the suggestions make way for the conversation.
    assert not [button for button in app.button if button.key and button.key.startswith("starter_")]


def test_a_rating_says_whether_it_was_saved():
    # A thumb click with no reply looks as if it did nothing.
    app = render_page(signed_in=True)
    answer_line = (
        'data: {"type": "final", "run_id": "run-1", "answer": "Five days.", '
        '"citations": [], "degraded": []}'
    )
    with patch("httpx.stream", return_value=make_streamed_response(event_lines=[answer_line])):
        app.chat_input[0].set_value("Hello").run()

    saved_reply = MagicMock()
    with patch("httpx.post", return_value=saved_reply):
        app.session_state["rating_run-1"] = 1
        app.run()

    assert [toast.value for toast in app.toast] == ["Thank you. Your rating was saved."]


def test_the_reduced_service_notice_uses_plain_sentences_and_keeps_the_ids():
    app = render_page(signed_in=True)
    raw_id = "supervisor_used_fallback_model:anthropic:claude-sonnet-5-5"
    answer_line = (
        'data: {"type": "final", "run_id": "run-2", "answer": "Five days.", "citations": [], '
        f'"degraded": ["{raw_id}"]}}'
    )

    with patch("httpx.stream", return_value=make_streamed_response(event_lines=[answer_line])):
        app.chat_input[0].set_value("Hello").run()

    assert [notice.value for notice in app.warning] == [
        "Reduced service: A backup AI model answered some steps."
    ]
    assert raw_id in [block.value for block in app.text]
