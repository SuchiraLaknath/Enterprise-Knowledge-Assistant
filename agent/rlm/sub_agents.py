"""The research sub-agents behind `delegate`: batching, recursion and checked findings.

Each sub-agent reads one small batch of sections and returns typed findings. Recursion happens
in `_analyse_batch`: work moves one level down when a batch is too large for one reader, or,
through `_follow_up_unresolved`, when a sub-agent reports that the section it was shown does not
hold the answer. Both stop at the depth limit.

Budget counters change only on the event loop, between awaits, so sub-agents running at once
cannot overspend (rule 7).
"""

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langsmith import traceable

from agent.dependencies import LlmClient
from agent.evidence_formatting import format_evidence_documents, make_prompt_safe
from agent.prompts.research import SUB_AGENT_REQUEST_TEMPLATE, SUB_AGENT_SYSTEM_PROMPT
from agent.schemas import BatchAnalysis, Finding
from api.events import EventType, ResearchStage, emit_event
from config.settings import ResearchLimits
from retrieval.models import RetrievedChunk

logger = structlog.get_logger()

# Map steps are many and simple, so they use the cheaper route.
SUB_AGENT_ROUTE = "fast"
# Depths count from one, so a `max_depth` of 2 means two levels of sub-agents.
TOP_LEVEL = 1

DEGRADED_BATCH_FAILED = "research_batch_failed"
DEGRADED_BATCH_TRUNCATED = "research_batch_truncated"
DEGRADED_BUDGET_REACHED = "research_budget_reached"
DEGRADED_DOCUMENTS_UNRESOLVED = "research_documents_unresolved"
DEGRADED_MODEL_FALLBACK = "research_used_fallback_model"


class ResearchBudgetExceeded(Exception):
    """A plan asked for more than the research limits allow (rule 7)."""


def _trace_sub_agent_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep document text out of the trace, showing only which chunks the sub-agent read."""
    return {
        "task": inputs["task"],
        "depth": inputs["depth"],
        "chunk_ids": [chunk.chunk_id for chunk in inputs["chunks"]],
    }


class SubAgentTeam:
    """Every sub-agent of one plan attempt, with the budget they share and what they found."""

    def __init__(
        self,
        llm: LlmClient,
        limits: ResearchLimits,
        read_document_chunks: Callable[[list[str]], Awaitable[list[RetrievedChunk]]],
        note_degraded: Callable[..., None],
        sub_agent_calls_used: int,
    ) -> None:
        """Take the session's reading and reporting, so that access stays checked in one place.

        Calls made by earlier plan attempts carry over, because that budget is per run.
        """
        self._llm = llm
        self._limits = limits
        self._read_document_chunks = read_document_chunks
        self._note_degraded = note_degraded
        self._sub_agent_gate = asyncio.Semaphore(limits.max_parallel_sub_agents)
        self._analysed_chunk_ids: set[str] = set()
        # Every document handed to a sub-agent, so that a report can say what was left out.
        self.documents_handed_on: set[str] = set()
        self._unresolved_at_deepest_level: set[str] = set()
        self._batches_by_depth: Counter[int] = Counter()
        self.sub_agent_calls = sub_agent_calls_used
        self.depth_reached = 0
        self.findings: list[Finding] = []

    @property
    def chunks_read(self) -> int:
        """Return how many chunks reached a sub-agent, for the panel's "read of total" line."""
        return len(self._analysed_chunk_ids)

    async def analyse_batches(self, task: str, chunk_batches: list[list[RetrievedChunk]]) -> None:
        """Check a plan's request against the budget, and fan out to sub-agents."""
        # A plan that asks for too much is a bad plan, so it fails rather than quietly shrinking.
        batches_requested = self._batches_by_depth[TOP_LEVEL] + len(chunk_batches)
        if batches_requested > self._limits.max_batches_per_level:
            raise ResearchBudgetExceeded(
                f"delegate accepts at most {self._limits.max_batches_per_level} batches, "
                f"but {batches_requested} were requested."
            )

        for chunks in chunk_batches:
            self.documents_handed_on.update(chunk.doc_id for chunk in chunks)

        emit_event(
            EventType.RESEARCH,
            stage=ResearchStage.BATCHES_PLANNED,
            depth=TOP_LEVEL,
            batch_sizes=[len(chunks) for chunks in chunk_batches],
        )
        await asyncio.gather(
            *[self._analyse_batch(task, chunks, TOP_LEVEL) for chunks in chunk_batches]
        )
        self._report_unresolved_documents()

    def _report_unresolved_documents(self) -> None:
        """Warn about documents no sub-agent covered, once every batch has finished.

        This waits for the end because one document can sit in two batches running at once, so a
        sub-agent that sees nothing for it does not mean that the other one will not.
        """
        found_doc_ids = {finding.doc_id for finding in self.findings}
        if self._unresolved_at_deepest_level - found_doc_ids:
            self._note_degraded(DEGRADED_DOCUMENTS_UNRESOLVED)

    async def _analyse_batch(self, task: str, chunks: list[RetrievedChunk], depth: int) -> None:
        """Analyse one batch, moving to a level below when it needs more than one reader."""
        if not self._reserve_batch(depth):
            return
        self.depth_reached = max(self.depth_reached, depth)

        # A batch too large for one reader is shared between two sub-agents one level down.
        if len(chunks) > self._limits.sub_agent_chunk_limit:
            if depth < self._limits.max_depth:
                middle = len(chunks) // 2
                await asyncio.gather(
                    self._analyse_batch(task, chunks[:middle], depth + 1),
                    self._analyse_batch(task, chunks[middle:], depth + 1),
                )
                return
            # At the deepest level there is nobody to pass it to, so the rest is left unread.
            chunks = chunks[: self._limits.sub_agent_chunk_limit]
            self._note_degraded(DEGRADED_BATCH_TRUNCATED)

        analysis = await self._call_sub_agent(task, chunks, depth)
        if analysis is None:
            return
        self._record_findings(analysis, chunks, depth)
        await self._follow_up_unresolved(task, analysis, chunks, depth)

    async def _follow_up_unresolved(
        self, task: str, analysis: BatchAnalysis, chunks: list[RetrievedChunk], depth: int
    ) -> None:
        """Take a second, fuller look at documents this sub-agent could not settle.

        That covers what it reported as unresolved, and any document it silently skipped, because
        a summary that quietly drops a document is worse than one that takes a little longer.
        """
        flagged_doc_ids = {
            chunks[marker - 1].doc_id
            for marker in analysis.unresolved_markers
            if 1 <= marker <= len(chunks)
        }
        found_doc_ids = {finding.doc_id for finding in self.findings}
        skipped_doc_ids = {chunk.doc_id for chunk in chunks} - found_doc_ids
        unresolved_doc_ids = sorted((flagged_doc_ids | skipped_doc_ids) - found_doc_ids)
        if not unresolved_doc_ids:
            return
        if depth >= self._limits.max_depth:
            self._unresolved_at_deepest_level.update(unresolved_doc_ids)
            return

        # All of each document's sections, because the one already shown was not enough.
        document_chunks = await self._read_document_chunks(unresolved_doc_ids)
        if document_chunks:
            await self._analyse_batch(task, document_chunks, depth + 1)

    def _reserve_batch(self, depth: int) -> bool:
        """Count one batch at a level, or return False if that level is full (rule 7)."""
        if self._batches_by_depth[depth] >= self._limits.max_batches_per_level:
            self._note_degraded(DEGRADED_BUDGET_REACHED)
            return False
        self._batches_by_depth[depth] += 1
        return True

    @traceable(name="research_sub_agent", run_type="chain", process_inputs=_trace_sub_agent_inputs)
    async def _call_sub_agent(
        self, task: str, chunks: list[RetrievedChunk], depth: int
    ) -> BatchAnalysis | None:
        """Ask one sub-agent about one batch. `depth` is here for the trace, which records it.

        Failures are data: one failed batch is recorded, and the others carry on. The budget
        check stays before the first await, so sub-agents running at once cannot overspend.
        """
        if self.sub_agent_calls >= self._limits.max_sub_agent_calls:
            self._note_degraded(DEGRADED_BUDGET_REACHED)
            return None
        self.sub_agent_calls += 1

        # Rule 5: the sections reach the sub-agent as data, inside document tags. A plan can
        # build its task from a title, so the task gets the same treatment.
        request_text = SUB_AGENT_REQUEST_TEMPLATE.format(
            task=make_prompt_safe(task), documents=format_evidence_documents(chunks)
        )
        messages: list[BaseMessage] = [
            SystemMessage(SUB_AGENT_SYSTEM_PROMPT),
            HumanMessage(request_text),
        ]
        async with self._sub_agent_gate:
            try:
                analysis_result = await self._llm.generate_structured(
                    SUB_AGENT_ROUTE, messages, BatchAnalysis
                )
            except Exception as error:  # One failed sub-agent must not fail the whole run.
                logger.warning("research_sub_agent_failed", error=type(error).__name__)
                self._note_degraded(DEGRADED_BATCH_FAILED)
                return None

        self._note_degraded(*analysis_result.describe_fallbacks(DEGRADED_MODEL_FALLBACK))
        self._analysed_chunk_ids.update(batch_chunk.chunk_id for batch_chunk in chunks)
        return analysis_result.value

    def _record_findings(
        self, analysis: BatchAnalysis, chunks: list[RetrievedChunk], depth: int
    ) -> None:
        """Keep each finding whose evidence is real, and report the batch to the panel.

        Rule 4: markers are checked against the batch this sub-agent was really shown, so a
        made-up number can never become a citation.
        """
        new_finding_count = 0
        for extracted_finding in analysis.findings:
            evidence_chunks = [
                chunks[marker - 1]
                for marker in extracted_finding.evidence_markers
                if 1 <= marker <= len(chunks)
            ]
            if not evidence_chunks:
                continue
            # A finding speaks for one document, or the theme counts would undercount.
            first_chunk = evidence_chunks[0]
            evidence_chunks = [
                evidence_chunk
                for evidence_chunk in evidence_chunks
                if evidence_chunk.doc_id == first_chunk.doc_id
            ]
            self.findings.append(
                Finding(
                    finding_id=f"F{len(self.findings) + 1}",
                    doc_id=first_chunk.doc_id,
                    title=first_chunk.title,
                    created_date=first_chunk.created_date,
                    claim=extracted_finding.claim,
                    theme_label=extracted_finding.theme_label,
                    evidence_chunk_ids=[
                        evidence_chunk.chunk_id for evidence_chunk in evidence_chunks
                    ],
                )
            )
            new_finding_count += 1

        emit_event(
            EventType.RESEARCH,
            stage=ResearchStage.BATCH_FINISHED,
            depth=depth,
            batch_size=len(chunks),
            finding_count=new_finding_count,
            chunks_read=self.chunks_read,
            sub_agent_calls=self.sub_agent_calls,
        )
