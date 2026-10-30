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


class Citation(BaseModel):
    """A source shown beside an answer, linked to its `[n]` marker."""

    marker: int
    chunk_id: str
    title: str
    section: str
    source_uri: str
