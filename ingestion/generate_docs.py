"""Generate the mock corpus as Markdown files with YAML front-matter in `data/docs/`.

Generation uses templates and a fixed random seed instead of an LLM, so anyone can reproduce
the same corpus without an API key, and search results stay comparable between runs.

Run with: uv run python -m ingestion.generate_docs
"""

import random
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

from config.settings import DOCUMENTS_DIRECTORY
from ingestion.corpus_content import (
    DOCUMENT_BRIEFS,
    EXTENDED_SECTION_TEMPLATES,
    INCIDENT_REVIEW_SECTIONS,
    NON_PAYMENT_INCIDENTS,
    PAYMENT_SERVICES,
    ROOT_CAUSES,
    SECTION_TEMPLATES,
    DocumentBrief,
    RootCauseProfile,
)

# A fixed seed and date keep the corpus identical on every run.
RANDOM_SEED = 2026
CORPUS_REFERENCE_DATE = date(2026, 10, 30)

# Payment incidents inside the last 12 months, by root cause.
RECENT_INCIDENT_COUNTS = {
    "certificate_expiry": 8,
    "connection_pool_exhaustion": 8,
    "card_scheme_gateway_timeout": 6,
    "scheduled_batch_delay": 2,
}
# Older incidents that a "last year" date filter must exclude.
OLDER_INCIDENT_AGES_IN_DAYS = (400, 430, 470)
OLDER_INCIDENT_ROOT_CAUSES = (
    "certificate_expiry",
    "connection_pool_exhaustion",
    "card_scheme_gateway_timeout",
)

RECENT_INCIDENT_MINIMUM_AGE_IN_DAYS = 12
RECENT_INCIDENT_MAXIMUM_AGE_IN_DAYS = 350

# Plausible ranges for the invented incident figures.
OUTAGE_MINUTES_RANGE = (18, 240)
FAILED_TRANSACTIONS_RANGE = (800, 42000)

# Non-payment incidents are spaced out, starting a few weeks before the reference date.
FIRST_NON_PAYMENT_INCIDENT_AGE_IN_DAYS = 50
DAYS_BETWEEN_NON_PAYMENT_INCIDENTS = 30

# Every fifth incident is confidential and every twelfth is restricted, so that users with
# different roles receive visibly different research results.
CONFIDENTIAL_INCIDENT_INTERVAL = 5
RESTRICTED_INCIDENT_INTERVAL = 12


@dataclass(frozen=True)
class GeneratedDocument:
    """One finished document: its metadata, and its body as Markdown."""

    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: date
    body: str

    @property
    def source_uri(self) -> str:
        """Return the citation link shown beside an answer."""
        return f"docs://{self.department}/{self.doc_id}"

    def to_markdown(self) -> str:
        """Return the file content: YAML front-matter followed by the body."""
        front_matter = {
            "doc_id": self.doc_id,
            "title": self.title,
            "department": self.department,
            "document_type": self.document_type,
            "access_level": self.access_level,
            "created_date": self.created_date.isoformat(),
            "source_uri": self.source_uri,
        }
        return f"---\n{yaml.safe_dump(front_matter, sort_keys=False)}---\n\n{self.body}\n"


def _join_sections(title: str, sections: list[tuple[str, str]]) -> str:
    """Render a title and (heading, paragraph) pairs as Markdown."""
    rendered_sections = [f"## {heading}\n\n{paragraph}" for heading, paragraph in sections]
    return "\n\n".join([f"# {title}", *rendered_sections])


def _build_incident_review_sections(service_name: str) -> list[tuple[str, str]]:
    """Return the review sections that every incident report shares."""
    return [
        (heading, paragraph.format(service=service_name))
        for heading, paragraph in INCIDENT_REVIEW_SECTIONS
    ]


def _incident_access_level(incident_number: int) -> str:
    """Spread access levels across incidents, with restricted taking priority."""
    if incident_number % RESTRICTED_INCIDENT_INTERVAL == 0:
        return "restricted"
    if incident_number % CONFIDENTIAL_INCIDENT_INTERVAL == 0:
        return "confidential"
    return "internal"


def _build_payment_incident(
    incident_number: int,
    root_cause_name: str,
    age_in_days: int,
    random_source: random.Random,
) -> GeneratedDocument:
    """Write one payment incident report from its root cause profile."""
    profile: RootCauseProfile = ROOT_CAUSES[root_cause_name]
    service = random_source.choice(PAYMENT_SERVICES)
    created_date = CORPUS_REFERENCE_DATE - timedelta(days=age_in_days)
    doc_id = f"INC-{created_date.year}-{incident_number:03d}"
    outage_minutes = random_source.randint(*OUTAGE_MINUTES_RANGE)
    failed_transactions = random_source.randint(*FAILED_TRANSACTIONS_RANGE)

    # The phrases below read as one report, and the seeded choice varies them between incidents.
    detection_signal = random_source.choice(profile.detection_signals)
    root_cause_statement = random_source.choice(profile.root_cause_statements)
    resolution_statement = random_source.choice(profile.resolution_statements)
    follow_up_action = random_source.choice(profile.follow_up_actions)

    sections = [
        (
            "Summary",
            f"{service} was degraded for {outage_minutes} minutes on "
            f"{created_date.isoformat()}. The incident was logged with error code "
            f"{profile.error_code} and handled as a major incident under the payment incident "
            "management policy. This report records what happened, why it happened, and what "
            "will change so that it does not happen again.",
        ),
        (
            "Customer impact",
            f"Approximately {failed_transactions} transactions failed or were delayed during the "
            "incident. Customers saw declined or pending payments, and the contact centre "
            "received more calls than usual. No customer funds were lost, and affected "
            "transactions were reconciled after the service recovered. Merchants were told "
            "through the usual partner channel.",
        ),
        (
            "Detection and timeline",
            f"{detection_signal} The on-call engineer acknowledged the alert within five minutes "
            "and opened a major incident. The incident lead joined the bridge and began sending "
            "status updates every 30 minutes. The service was declared stable after 15 minutes "
            "of normal behaviour on the dashboard.",
        ),
        (
            "Root cause",
            f"{root_cause_statement} The review found that the warning signs were visible before "
            "customers were affected, but no alert was set up to act on them.",
        ),
        (
            "Resolution",
            f"{resolution_statement} The team confirmed recovery with synthetic transactions, "
            "and the service desk closed the customer communication once the backlog cleared.",
        ),
        (
            "Follow-up actions",
            f"{follow_up_action} The action has a named owner and a due date, and it is tracked "
            "in the payments reliability review until it is closed.",
        ),
        *_build_incident_review_sections(service),
    ]
    title = f"{service} outage {created_date.isoformat()}"
    return GeneratedDocument(
        doc_id=doc_id,
        title=title,
        department="payments",
        document_type="incident",
        access_level=_incident_access_level(incident_number),
        created_date=created_date,
        body=_join_sections(title, sections),
    )


def _plan_payment_incidents(random_source: random.Random) -> list[tuple[str, int]]:
    """Return (root cause, age in days) pairs, spread evenly over the last 12 months."""
    recent_root_causes = [
        root_cause_name
        for root_cause_name, incident_count in RECENT_INCIDENT_COUNTS.items()
        for _ in range(incident_count)
    ]
    random_source.shuffle(recent_root_causes)

    # Equal spacing puts incidents across the whole year, with none on the same day.
    age_range = RECENT_INCIDENT_MAXIMUM_AGE_IN_DAYS - RECENT_INCIDENT_MINIMUM_AGE_IN_DAYS
    days_between_incidents = age_range // (len(recent_root_causes) - 1)
    planned_incidents = [
        (root_cause_name, RECENT_INCIDENT_MINIMUM_AGE_IN_DAYS + position * days_between_incidents)
        for position, root_cause_name in enumerate(recent_root_causes)
    ]
    older_incidents = zip(OLDER_INCIDENT_ROOT_CAUSES, OLDER_INCIDENT_AGES_IN_DAYS, strict=True)
    planned_incidents.extend(older_incidents)
    return planned_incidents


def _build_non_payment_incidents() -> list[GeneratedDocument]:
    """Write the few non-payment incidents, which should not surface in payment research."""
    documents = []
    for position, (department, title, cause) in enumerate(NON_PAYMENT_INCIDENTS):
        age_in_days = (
            FIRST_NON_PAYMENT_INCIDENT_AGE_IN_DAYS + position * DAYS_BETWEEN_NON_PAYMENT_INCIDENTS
        )
        created_date = CORPUS_REFERENCE_DATE - timedelta(days=age_in_days)
        sections = [
            (
                "Summary",
                f"This report covers an incident in the {department} department: {title}. It "
                "records the cause, the response and the improvements that were agreed.",
            ),
            (
                "Root cause",
                f"{cause} The issue was found through routine monitoring and was contained "
                "without any effect on customer payments.",
            ),
            (
                "Resolution",
                "The owning team corrected the configuration and confirmed normal behaviour. "
                "The incident lead reviewed the timeline with everyone involved.",
            ),
            (
                "Follow-up actions",
                "The related runbook was updated, and the alert thresholds were reviewed so that "
                "the same condition is caught earlier next time.",
            ),
            *_build_incident_review_sections(f"the {department} service"),
        ]
        documents.append(
            GeneratedDocument(
                doc_id=f"INC-OPS-{position + 1:03d}",
                title=title,
                department=department,
                document_type="incident",
                access_level="internal",
                created_date=created_date,
                body=_join_sections(title, sections),
            )
        )
    return documents


def _build_document_from_brief(brief: DocumentBrief) -> GeneratedDocument:
    """Fill the section templates for a document type with one brief's facts."""
    fact_1, fact_2, fact_3 = brief.key_facts
    section_templates = (
        *SECTION_TEMPLATES[brief.document_type],
        *EXTENDED_SECTION_TEMPLATES[brief.document_type],
    )
    sections = []
    for template in section_templates:
        filled_sentences = [
            sentence.format(subject=brief.subject, fact_1=fact_1, fact_2=fact_2, fact_3=fact_3)
            for sentence in template.sentences
        ]
        sections.append((template.heading, " ".join(filled_sentences)))
    return GeneratedDocument(
        doc_id=brief.doc_id,
        title=brief.title,
        department=brief.department,
        document_type=brief.document_type,
        access_level=brief.access_level,
        created_date=CORPUS_REFERENCE_DATE - timedelta(days=brief.age_in_days),
        body=_join_sections(brief.title, sections),
    )


def build_corpus() -> list[GeneratedDocument]:
    """Build every document in memory, so that tests can check the corpus without files."""
    random_source = random.Random(RANDOM_SEED)
    planned_incidents = _plan_payment_incidents(random_source)
    payment_incidents = [
        _build_payment_incident(incident_number, root_cause_name, age_in_days, random_source)
        for incident_number, (root_cause_name, age_in_days) in enumerate(planned_incidents, start=1)
    ]
    documents_from_briefs = [_build_document_from_brief(brief) for brief in DOCUMENT_BRIEFS]
    return [*payment_incidents, *_build_non_payment_incidents(), *documents_from_briefs]


def write_corpus(documents: list[GeneratedDocument], target_directory: Path) -> None:
    """Write each document to `<target_directory>/<department>/<doc_id>.md`.

    The directory is emptied first, because this folder holds only generated files, and a
    stale file from a moved document would return at its old access level.
    """
    shutil.rmtree(target_directory, ignore_errors=True)
    for document in documents:
        department_directory = target_directory / document.department
        department_directory.mkdir(parents=True, exist_ok=True)
        document_path = department_directory / f"{document.doc_id}.md"
        document_path.write_text(document.to_markdown(), encoding="utf-8")


if __name__ == "__main__":
    corpus = build_corpus()
    write_corpus(corpus, DOCUMENTS_DIRECTORY)
    print(f"Wrote {len(corpus)} documents to {DOCUMENTS_DIRECTORY}")
