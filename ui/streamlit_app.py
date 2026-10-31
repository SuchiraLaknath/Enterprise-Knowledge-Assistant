"""Streamlit chat with a live view of what the agent is doing.

The brief asks for function and transparency over looks, so this stays simple: sign in on the
left, chat in the middle and the activity panel on the right. The message box stays at the
bottom of the window, and the two panels above it scroll on their own, as in other chat apps.

Run with: uv run streamlit run ui/streamlit_app.py
"""

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Streamlit runs this file as a script, so the project root must be added to find `config` and `ui`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_environment_settings  # noqa: E402
from ui.demo_users import DEMO_USERS, format_user_label  # noqa: E402
from ui.event_formatting import (  # noqa: E402
    TurnStatus,
    add_activity_event,
    apply_token_event,
    describe_degraded,
    describe_turn_status,
    parse_sse_line,
    update_turn_status,
)

REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)

# `st.feedback` reports thumbs as positions, where 1 is thumbs up.
THUMBS_UP_POSITION = 1
UNKNOWN_QUESTION = "(not recorded)"

MESSAGE_BOX_HINT = "Ask about policies, runbooks, incidents and more"
APPROVAL_WAITING_HINT = "Approve or reject the action first"
API_UNREACHABLE_MESSAGE = "The assistant could not be reached. Please try again."
NO_ACTIVITY_MESSAGE = "Each step of the agent appears here."
NO_ANSWER_MESSAGE = "No answer arrived. Please try again."
SESSION_EXPIRED_MESSAGE = "Your session has expired. Please sign in again."
# Used only inside this file, between reading the API's reply and drawing the page.
SESSION_EXPIRED_EVENT = "session_expired"

# Streamlit gives container keys to the page as `st-key-<key>` classes, which the style below uses.
CONVERSATION_PANEL_KEY = "conversation_panel"
ACTIVITY_PANEL_KEY = "activity_panel"
# Streamlit only scrolls and follows new text in a container with a height in pixels, so the
# panels are created with this height, and the style below replaces it to fit the window.
PANEL_HEIGHT_PIXELS = 480
# The height leaves room for the panel titles and the message box, so the page itself never
# scrolls and the message box stays in view. Streamlit sets the height on the wrapper around each
# panel, so the rule targets that wrapper. The text here is fixed and holds no user content.
PANEL_LAYOUT_STYLE = f"""
<style>
.stMainBlockContainer {{ padding-top: 3.75rem !important; padding-bottom: 0 !important; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-{CONVERSATION_PANEL_KEY}),
[data-testid="stLayoutWrapper"]:has(> .st-key-{ACTIVITY_PANEL_KEY}) {{
    flex-basis: calc(100dvh - 16rem) !important;
    height: calc(100dvh - 16rem) !important;
}}
</style>
"""


def api_url(path: str) -> str:
    """Return the full address of an API route."""
    return f"{get_environment_settings().api_url}{path}"


def auth_headers() -> dict[str, str]:
    """Return the header that proves who is signed in."""
    return {"Authorization": f"Bearer {st.session_state.access_token}"}


def start_new_conversation() -> None:
    """Clear the chat and activity, and use a new thread."""
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.chat_history = []
    st.session_state.activity_lines = []
    st.session_state.turn_status = TurnStatus()
    st.session_state.pop("pending_approval", None)


def end_session_with_notice(notice: str) -> None:
    """Sign out and redraw the page, so that the sidebar never claims a dead session is live."""
    del st.session_state["access_token"]
    start_new_conversation()
    st.session_state.sign_in_notice = notice
    st.rerun()


def show_sign_in() -> None:
    """Show the sign-in controls, or the signed-in user."""
    if "sign_in_notice" in st.session_state:
        st.warning(st.session_state.pop("sign_in_notice"))

    if "access_token" in st.session_state:
        st.write(f"Signed in as **{format_user_label(st.session_state.username)}**")
        st.caption(DEMO_USERS[st.session_state.username].capabilities)
        if st.button("New conversation", width="stretch"):
            start_new_conversation()
            st.rerun()
        if st.button("Sign out", width="stretch"):
            del st.session_state["access_token"]
            start_new_conversation()
            st.rerun()
        return

    username = st.selectbox("Demo user", list(DEMO_USERS), format_func=format_user_label)
    st.caption(DEMO_USERS[username].capabilities)
    if st.button("Sign in", type="primary", width="stretch"):
        try:
            response = httpx.post(
                api_url("/auth/login"),
                json={"username": username, "password": DEMO_USERS[username].password},
                timeout=REQUEST_TIMEOUT,
            )
        except httpx.HTTPError:
            st.error(API_UNREACHABLE_MESSAGE)
            return
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


def send_rating(final_event: dict[str, Any], thumbs_position: int) -> None:
    """Record a rating. The answer has already been given, so a failure only shows a note."""
    rating = "up" if thumbs_position == THUMBS_UP_POSITION else "down"
    payload = {
        "run_id": final_event["run_id"],
        "rating": rating,
        "question": final_event.get("question") or UNKNOWN_QUESTION,
    }
    try:
        httpx.post(
            api_url("/feedback"), json=payload, headers=auth_headers(), timeout=REQUEST_TIMEOUT
        )
    except httpx.HTTPError:
        st.caption("Your rating could not be saved.")


def show_answer_details(final_event: dict[str, Any]) -> None:
    """Show the degradation notice, the sources, the trace link and the rating buttons."""
    if final_event.get("degraded"):
        sentences = describe_degraded(final_event["degraded"])
        st.warning(f"Reduced service: {' '.join(sentences)}")
        # The exact ids stay one click away, because they are what an evaluator would look up.
        with st.expander("Technical details"):
            st.text("\n".join(final_event["degraded"]))
    show_citations(final_event.get("citations", []))
    if final_event.get("trace_url"):
        st.link_button("View LangSmith trace", final_event["trace_url"])
    # The key is the run id, so that each answer keeps its own rating across reruns.
    thumbs_position = st.feedback("thumbs", key=f"rating_{final_event.get('run_id')}")
    if thumbs_position is not None and not final_event.get("rating_sent"):
        send_rating(final_event, thumbs_position)
        final_event["rating_sent"] = True


def stream_events(path: str, payload: dict[str, Any]) -> Any:
    """Yield each event of one turn as it arrives from the API."""
    with httpx.stream(
        "POST",
        api_url(path),
        json=payload,
        headers=auth_headers(),
        timeout=REQUEST_TIMEOUT,
    ) as response:
        if response.status_code == httpx.codes.UNAUTHORIZED:
            yield {"type": SESSION_EXPIRED_EVENT}
            return
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            wait = response.headers.get("retry-after", "a few")
            yield {"type": "error", "message": f"Too many requests. Try again in {wait} seconds."}
            return
        if response.status_code >= httpx.codes.BAD_REQUEST:
            yield {"type": "error", "message": "The request could not be completed."}
            return
        for line in response.iter_lines():
            event_data = parse_sse_line(line)
            if event_data:
                yield json.loads(event_data)


@dataclass
class ActivityView:
    """The two places on the page that show what the agent is doing."""

    status_line: Any
    panel: Any

    def refresh(self) -> None:
        """Redraw both. Rule 9: this is plain text, as the lines can hold words from a model."""
        self.status_line.text(describe_turn_status(st.session_state.turn_status))
        activity_text = "\n".join(st.session_state.activity_lines)
        self.panel.code(activity_text or NO_ACTIVITY_MESSAGE, language=None, wrap_lines=True)


def show_streamed_turn(events: Any, activity_view: ActivityView, question: str) -> None:
    """Draw one turn as it arrives. Tokens go to the draft, and `final` replaces them."""
    error_was_shown = False
    with st.chat_message("assistant"):
        answer_area = st.empty()
        draft = ""
        final_event: dict[str, Any] | None = None

        try:
            for event in events:
                update_turn_status(st.session_state.turn_status, event)
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
                elif event["type"] == SESSION_EXPIRED_EVENT:
                    end_session_with_notice(SESSION_EXPIRED_MESSAGE)
                elif event["type"] == "error":
                    st.error(event["message"])
                    error_was_shown = True
                else:
                    if event["type"] == "approval":
                        st.session_state.pending_approval = event
                    add_activity_event(st.session_state.activity_lines, event)
                activity_view.refresh()
        except httpx.HTTPError:
            st.error(API_UNREACHABLE_MESSAGE)
            error_was_shown = True

        # Rule 9: only the final answer is rendered as markdown, and a turn that never finished
        # must not leave an unconfirmed draft on the screen.
        if final_event is None:
            answer_area.empty()
            # A pause is not a failure, because the approval box comes next.
            if not error_was_shown and "pending_approval" not in st.session_state:
                st.error(NO_ANSWER_MESSAGE)
        else:
            final_event["question"] = question
            show_answer_details(final_event)
            st.session_state.chat_history.append(
                {"role": "assistant", "text": final_event["answer"], "final_event": final_event}
            )


def run_chat_turn(message: str, activity_view: ActivityView) -> None:
    """Send a message, and update the draft and the activity panel as events arrive."""
    st.session_state.chat_history.append({"role": "user", "text": message})
    st.session_state.activity_lines = []
    st.session_state.turn_status = TurnStatus()
    st.session_state.last_question = message
    with st.chat_message("user"):
        st.text(message)
    payload = {"message": message, "thread_id": st.session_state.thread_id}
    show_streamed_turn(stream_events("/chat", payload), activity_view, message)


def run_approval_turn(approved: bool, activity_view: ActivityView) -> None:
    """Send a decision on a paused write tool, and show the rest of the turn."""
    pending_call = st.session_state.pop("pending_approval")
    payload = {
        "thread_id": st.session_state.thread_id,
        "tool_call_id": pending_call["tool_call_id"],
        "approved": approved,
    }
    question = st.session_state.get("last_question", "")
    # A resumed turn carries on with the same activity, so that the whole story stays visible.
    st.session_state.turn_status.is_finished = False
    show_streamed_turn(stream_events("/approve", payload), activity_view, question)


def show_pending_approval(activity_view: ActivityView) -> None:
    """Ask the person to decide on a paused write tool, showing exactly what would run."""
    pending_call = st.session_state.get("pending_approval")
    if pending_call is None:
        return
    with st.chat_message("assistant"):
        st.warning("This action changes data, and needs your approval.")
        # Plain text, because a tool's arguments came from a model.
        st.text(f"{pending_call['tool_name']}\n{json.dumps(pending_call['arguments'], indent=2)}")
        approve_column, reject_column = st.columns(2)
        if approve_column.button("Approve", key="approve_tool_call"):
            run_approval_turn(True, activity_view)
            st.rerun()
        if reject_column.button("Reject", key="reject_tool_call"):
            run_approval_turn(False, activity_view)
            st.rerun()


def show_chat_history() -> None:
    """Redraw earlier messages, which Streamlit clears on every rerun."""
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.text(entry["text"])
            else:
                st.markdown(entry["text"])
                show_answer_details(entry["final_event"])


def choose_starter_question(question: str) -> None:
    """Keep the chosen question for the rerun that a button click starts."""
    st.session_state.starter_question = question


def show_starter_questions() -> None:
    """Fill an empty chat with questions that show what this user can do."""
    st.write("Try one of these questions, or type your own below.")
    for question in DEMO_USERS[st.session_state.username].starter_questions:
        st.button(
            question,
            key=f"starter_{question}",
            on_click=choose_starter_question,
            args=(question,),
            width="stretch",
        )


def main() -> None:
    """Lay out the page and handle one interaction."""
    st.set_page_config(page_title="Knowledge Assistant", layout="wide")
    st.html(PANEL_LAYOUT_STYLE)
    if "thread_id" not in st.session_state:
        start_new_conversation()

    with st.sidebar:
        st.title("Knowledge Assistant")
        st.caption("Commercial Bank internal assistant. All data is mock data.")
        show_sign_in()

    chat_column, activity_column = st.columns([3, 2])
    with activity_column:
        st.subheader("Agent activity")
        status_line = st.empty()
        with st.container(height=PANEL_HEIGHT_PIXELS, key=ACTIVITY_PANEL_KEY, autoscroll=True):
            activity_view = ActivityView(status_line, st.empty())
        activity_view.refresh()

    with chat_column:
        st.subheader("Conversation")
        conversation_panel = st.container(
            height=PANEL_HEIGHT_PIXELS, key=CONVERSATION_PANEL_KEY, autoscroll=True
        )

    if "access_token" not in st.session_state:
        conversation_panel.info("Sign in from the sidebar to start.")
        return

    # This stays outside the columns, because Streamlit pins only a top-level message box to the
    # bottom of the window. Inside a column, it would sit in the page with the messages.
    # A new message during a pause would leave the paused tool call undecided.
    waiting_for_decision = "pending_approval" in st.session_state
    message = st.chat_input(
        APPROVAL_WAITING_HINT if waiting_for_decision else MESSAGE_BOX_HINT,
        disabled=waiting_for_decision,
    )
    message = message or st.session_state.pop("starter_question", None)
    with conversation_panel:
        show_chat_history()
        if message:
            run_chat_turn(message, activity_view)
            # The turn may have paused, and the message box above was drawn before it did.
            if "pending_approval" in st.session_state:
                st.rerun()
        elif not st.session_state.chat_history:
            show_starter_questions()
        show_pending_approval(activity_view)


main()
