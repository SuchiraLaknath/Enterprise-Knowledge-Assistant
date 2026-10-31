from pathlib import Path

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
