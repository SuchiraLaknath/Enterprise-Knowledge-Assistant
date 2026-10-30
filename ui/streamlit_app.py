"""Streamlit chat with a live view of what the agent is doing.

The brief asks for function and transparency over looks, so this stays simple: sign in on the
left, chat in the middle and the activity panel on the right.

Run with: uv run streamlit run ui/streamlit_app.py
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Streamlit runs this file as a script, so the project root must be added to find `config` and `ui`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_environment_settings  # noqa: E402
from ui.event_formatting import apply_token_event, describe_event, parse_sse_line  # noqa: E402

REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)
# The sign-in shortcuts for the demo. The API holds the real user list and checks these.
DEMO_USERS = {"vimal": "viewer-demo", "nadee": "analyst-demo", "admin": "admin-demo"}


def api_url(path: str) -> str:
    """Return the full address of an API route."""
    return f"{get_environment_settings().api_url}{path}"


def start_new_conversation() -> None:
    """Clear the chat and activity, and use a new thread."""
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.chat_history = []
    st.session_state.activity_lines = []


def show_sign_in() -> None:
    """Show the sign-in controls, or the signed-in user."""
    if "access_token" in st.session_state:
        st.write(f"Signed in as **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Sign out"):
            del st.session_state["access_token"]
            start_new_conversation()
            st.rerun()
        return

    username = st.selectbox("Demo user", list(DEMO_USERS))
    if st.button("Sign in"):
        response = httpx.post(
            api_url("/auth/login"),
            json={"username": username, "password": DEMO_USERS[username]},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == httpx.codes.OK:
            login = response.json()
            st.session_state.access_token = login["access_token"]
            st.session_state.username = login["username"]
            st.session_state.role = login["role"]
            start_new_conversation()
            st.rerun()
        else:
            st.error("Sign-in failed.")


def show_citations(citations: list[dict[str, Any]]) -> None:
    """List the sources behind an answer."""
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            st.text(
                f"[{citation['marker']}] {citation['title']} > {citation['section']}\n"
                f"    {citation['source_uri']}"
            )


def show_answer_details(final_event: dict[str, Any]) -> None:
    """Show the degradation notice, the sources and the trace link."""
    if final_event.get("degraded"):
        st.warning(f"Reduced service: {', '.join(final_event['degraded'])}")
    show_citations(final_event.get("citations", []))
    if final_event.get("trace_url"):
        st.link_button("View LangSmith trace", final_event["trace_url"])


def stream_events(message: str) -> Any:
    """Yield each event of one chat turn as it arrives from the API."""
    with httpx.stream(
        "POST",
        api_url("/chat"),
        json={"message": message, "thread_id": st.session_state.thread_id},
        headers={"Authorization": f"Bearer {st.session_state.access_token}"},
        timeout=REQUEST_TIMEOUT,
    ) as response:
        if response.status_code == httpx.codes.UNAUTHORIZED:
            yield {"type": "error", "message": "Your session has expired. Please sign in again."}
            return
        for line in response.iter_lines():
            event_data = parse_sse_line(line)
            if event_data:
                yield json.loads(event_data)


def run_chat_turn(message: str, activity_panel: Any) -> None:
    """Send a message, and update the draft and the activity panel as events arrive."""
    st.session_state.chat_history.append({"role": "user", "text": message})
    st.session_state.activity_lines = []
    with st.chat_message("user"):
        st.text(message)

    with st.chat_message("assistant"):
        answer_area = st.empty()
        draft = ""
        final_event: dict[str, Any] | None = None

        try:
            # Tokens go to the draft, `final` replaces it, and the rest goes to the panel.
            for event in stream_events(message):
                if event["type"] == "token":
                    draft = apply_token_event(draft, event)
                    # Rule 9: a draft is plain text, so an injected image URL can never load.
                    answer_area.text(draft)
                elif event["type"] == "final":
                    final_event = event
                    # Show the answer now, rather than after the trace link arrives.
                    answer_area.markdown(final_event["answer"])
                elif event["type"] == "trace" and final_event is not None:
                    final_event["trace_url"] = event["url"]
                elif event["type"] == "error":
                    st.error(event["message"])
                else:
                    activity_line = describe_event(event)
                    if activity_line:
                        st.session_state.activity_lines.append(activity_line)
                        activity_panel.text("\n".join(st.session_state.activity_lines))
        except httpx.HTTPError:
            st.error("The assistant could not be reached. Please try again.")

        # Rule 9: only the final answer is rendered as markdown, and a turn that never finished
        # must not leave an unconfirmed draft on the screen.
        if final_event is None:
            answer_area.empty()
        else:
            show_answer_details(final_event)
            st.session_state.chat_history.append(
                {"role": "assistant", "text": final_event["answer"], "final_event": final_event}
            )


def show_chat_history() -> None:
    """Redraw earlier messages, which Streamlit clears on every rerun."""
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.text(entry["text"])
            else:
                st.markdown(entry["text"])
                show_answer_details(entry["final_event"])


def main() -> None:
    """Lay out the page and handle one interaction."""
    st.set_page_config(page_title="Knowledge Assistant", layout="wide")
    if "thread_id" not in st.session_state:
        start_new_conversation()

    with st.sidebar:
        st.title("Knowledge Assistant")
        st.caption("Commercial Bank internal assistant. All data is mock data.")
        show_sign_in()
        if "access_token" in st.session_state and st.button("New conversation"):
            start_new_conversation()
            st.rerun()

    chat_column, activity_column = st.columns([3, 2])
    with activity_column:
        st.subheader("Agent activity")
        activity_panel = st.empty()
        activity_panel.text("\n".join(st.session_state.activity_lines) or "Waiting for a question.")

    with chat_column:
        if "access_token" not in st.session_state:
            st.info("Sign in from the sidebar to start.")
            return
        show_chat_history()
        message = st.chat_input("Ask about policies, runbooks, incidents and more")
        if message:
            run_chat_turn(message, activity_panel)


main()
