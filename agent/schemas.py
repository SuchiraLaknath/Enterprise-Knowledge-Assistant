"""Typed records exchanged between agents, including structured LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field

Department = Literal["payments", "platform", "security", "hr-policy", "product"]
DocumentType = Literal[
    "policy", "architecture", "runbook", "incident", "product_spec", "meeting_notes"
]


class Plan(BaseModel):
    """The supervisor's understanding of a request and how to handle it.

    It can narrow a search, but it has no access-level field: the LLM never decides access.
    """

    intent: str = Field(description="One sentence stating what the user wants to know.")
    steps: list[str] = Field(description="The short ordered steps needed to answer.")
    needs_documents: bool = Field(
        description="True unless this is a greeting or a question about the conversation itself."
    )
    search_query: str = Field(
        description="A standalone search query, with pronouns resolved from the conversation."
    )
    departments: list[Department] = Field(
        description="Departments to search. Empty means search all of them."
    )
    document_types: list[DocumentType] = Field(
        description="Document types to search. Empty means search all of them."
    )
    recent_days: int | None = Field(
        description="Limit to documents from the last N days, or null for no date limit."
    )
    needs_research: bool = Field(
        description=(
            "True if answering needs many documents read and compared, such as summarising all "
            "reports on a topic or finding recurring causes. False for one specific fact."
        )
    )
    needs_tools: bool = Field(
        description=(
            "True if answering needs live data from a tool: people, services, current incidents, "
            "the audit log, or a calculation over what was found."
        )
    )


class ResearchCode(BaseModel):
    """A Python search plan from the research planner. It runs only in the sandbox (rule 7)."""

    reasoning: str = Field(description="One or two sentences on how the plan answers the task.")
    python_code: str = Field(description="The plan, using only the allowed syntax and helpers.")


class ExtractedFinding(BaseModel):
    """What a research sub-agent says it found. Code checks its markers before keeping it."""

    claim: str = Field(description="One factual sentence stating what the document shows.")
    theme_label: str = Field(description="A short name for this kind of finding, in 2 to 5 words.")
    evidence_markers: list[int] = Field(
        description="The id numbers of the document tags that support this claim."
    )


class BatchAnalysis(BaseModel):
    """A research sub-agent's reply about one batch of document sections."""

    findings: list[ExtractedFinding] = Field(
        description="One finding per document that answers the task."
    )
    unresolved_markers: list[int] = Field(
        description=(
            "Id numbers of document tags that are about the task, but do not hold the answer in "
            "the text shown. Other sections of those documents are then read. Usually empty."
        )
    )


class Finding(BaseModel):
    """A checked finding. Rule 4: `evidence_chunk_ids` holds only chunks that were really read."""

    finding_id: str
    doc_id: str
    title: str
    created_date: str
    claim: str
    theme_label: str
    evidence_chunk_ids: list[str]


class ThemeDraft(BaseModel):
    """One theme as the aggregator proposes it. Code does all the counting."""

    name: str = Field(description="A short name for this theme.")
    summary: str = Field(description="One or two sentences describing what these findings share.")
    finding_ids: list[str] = Field(description="The ids of the findings that belong here.")


class ThemeGrouping(BaseModel):
    """The aggregator's grouping of findings into themes."""

    themes: list[ThemeDraft] = Field(description="The themes, most common first.")


class ResearchTheme(BaseModel):
    """A theme with its findings. Counts come from code, so a model cannot invent them."""

    name: str
    summary: str
    document_count: int
    first_date: str
    last_date: str
    findings: list[Finding]


class ResearchReport(BaseModel):
    """The result of one research run, which the response agent turns into an answer."""

    themes: list[ResearchTheme]
    documents_analysed: int
    chunks_read: int
    chunks_available: int
    depth_reached: int
    sub_agent_calls: int


class InputJudgement(BaseModel):
    """The input guard's LLM judge on one message."""

    is_attack: bool = Field(description="True if this message attacks the assistant itself.")
    category: str = Field(description="The kind of attack, or an empty string if none.")


class AnswerJudgement(BaseModel):
    """The validator's LLM judge on one answer."""

    supported_by_evidence: bool = Field(description="True if the evidence backs every claim.")
    on_brand: bool = Field(description="False if the answer breaks a brand rule.")
    problem: str = Field(description="One sentence on any failure, or an empty string.")


class ExtractedFacts(BaseModel):
    """Lasting facts an employee stated about themselves."""

    facts: list[str] = Field(description="Short facts in the third person. Usually empty.")


class ConversationSummary(BaseModel):
    """A rolling summary of the older part of a conversation."""

    summary: str = Field(description="A few sentences on what was asked and decided.")


class ValidationReport(BaseModel):
    """What the validator decided, and why."""

    status: Literal["pass", "fail", "insufficient_evidence", "replaced_with_excerpts"]
    reasons: list[str]


class Citation(BaseModel):
    """A source shown beside an answer, linked to its `[n]` marker."""

    marker: int
    chunk_id: str
    title: str
    section: str
    source_uri: str
