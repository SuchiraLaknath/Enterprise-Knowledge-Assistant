"""The demo users as the sign-in screen shows them.

Rule 1: everything here is for display and for one-click sign-in. The API holds the real user
list and looks up the role from the verified token on every request, so changing a role in this
file changes a label and gives nobody any extra access.
"""

from dataclasses import dataclass

DEMO_RESEARCH_QUESTION = (
    "Summarize all outage reports related to payment failures during the last year "
    "and identify recurring root causes."
)
ANNUAL_LEAVE_QUESTION = "How many days of annual leave can I carry over?"


@dataclass(frozen=True)
class DemoUser:
    """One demo user, and what the sign-in screen and empty chat show for them."""

    password: str
    role: str
    capabilities: str
    starter_questions: tuple[str, ...]


DEMO_USERS = {
    "vimal": DemoUser(
        password="viewer-demo",
        role="viewer",
        capabilities="Reads public and internal documents.",
        # The research question is here on purpose: it shows the polite limit note.
        starter_questions=(
            ANNUAL_LEAVE_QUESTION,
            "What is the phishing response process?",
            DEMO_RESEARCH_QUESTION,
        ),
    ),
    "nadee": DemoUser(
        password="analyst-demo",
        role="analyst",
        capabilities=(
            "Also reads confidential documents, and uses deep research, "
            "live tools and Python analysis."
        ),
        starter_questions=(
            DEMO_RESEARCH_QUESTION,
            "Who owns the Card Switch service, and which incidents are open right now?",
            ANNUAL_LEAVE_QUESTION,
        ),
    ),
    "admin": DemoUser(
        password="admin-demo",
        role="admin",
        capabilities=(
            "Also reads restricted documents, sees the audit log, "
            "and can update incidents with approval."
        ),
        starter_questions=(
            DEMO_RESEARCH_QUESTION,
            "Set the status of incident LIVE-1041 to resolved.",
            "Show the latest entries in the audit log.",
        ),
    ),
}


def format_user_label(username: str) -> str:
    """Return a name with its role, such as "nadee - analyst"."""
    return f"{username} - {DEMO_USERS[username].role}"
