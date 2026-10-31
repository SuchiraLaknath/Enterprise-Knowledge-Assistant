"""The helpers an RLM plan can call, and the sub-agents behind `delegate`.

A plan is code written by a model, so what it can reach matters. `ResearchSession` holds the
verified user and the document text, and a plan only ever receives ids, titles, sections and
dates. No helper takes a user or an access level, so a plan cannot ask for more than the
employee may read (rules 1 and 7).

Helpers are synchronous, because plans run in the sandbox thread, but search and LLM calls are
async. `_run_on_event_loop` hops back to the event loop for that work. Waiting there uses none
of the sandbox's CPU budget.

Recursion happens in `_analyse_batch`: a sub-agent's work moves one level down when a batch is
too large for one reader, or, through `_follow_up_unresolved`, when a sub-agent reports that the
section it was shown does not hold the answer. Both stop at the depth limit.
"""

import asyncio
import time
from collections import Counter
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langsmith import traceable

from agent.dependencies import AgentDependencies
from agent.evidence_formatting import format_evidence_documents
from agent.prompts.research import SUB_AGENT_REQUEST_TEMPLATE, SUB_AGENT_SYSTEM_PROMPT
from agent.schemas import BatchAnalysis, Finding, Plan
from api.events import EventType, ResearchStage, emit_event
from retrieval.filters import keep_permitted_chunks, timestamp_days_ago
from retrieval.models import DocumentFilters, RetrievedChunk, UserContext

logger = structlog.get_logger()

# Map steps are many and simple, so they use the cheaper route.
SUB_AGENT_ROUTE = "fast"
# Depths count from one, so a `max_depth` of 2 means two levels of sub-agents.
TOP_LEVEL = 1
# What a plan may see of a catalog entry. It holds no text, and `access_level` is not needed.
PLAN_VISIBLE_CATALOG_FIELDS = {
    "doc_id",
    "title",
    "department",
    "document_type",
    "created_date",
    "chunk_count",
}

DEGRADED_BATCH_FAILED = "research_batch_failed"
DEGRADED_BATCH_TRUNCATED = "research_batch_truncated"
DEGRADED_BUDGET_REACHED = "research_budget_reached"
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


def _describe_chunk(chunk: RetrievedChunk) -> dict[str, str]:
    """Return what a plan may see of a chunk, which never includes its text."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "created_date": chunk.created_date,
    }


def group_by_doc(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group section records by document, so that a plan can work per document."""
    records_by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_doc.setdefault(record["doc_id"], []).append(record)
    return records_by_doc


def chunk(items: Sequence[Any], size: int) -> list[list[Any]]:
    """Split a list into batches, so that each sub-agent gets a manageable share."""
    items_as_list = list(items)
    return [items_as_list[start : start + size] for start in range(0, len(items_as_list), size)]


class ResearchSession:
    """One research run: its helpers, its budget, and everything it has read and found.

    Each helper a plan calls is a plain method, and the async work it needs has the same name
    with a leading underscore. Budget counters change only on the event loop, between awaits, so
    sub-agents running at once cannot overspend. Search and batch limits apply to one plan, and
    sub-agent calls and the deadline apply to the whole run, across plan attempts.
    """

    def __init__(
        self,
        user: UserContext,
        dependencies: AgentDependencies,
        sub_agent_calls_used: int = 0,
        run_deadline: float | None = None,
    ) -> None:
        """Start a plan attempt, carrying over what earlier attempts of this run used."""
        self._user = user
        self._dependencies = dependencies
        self._limits = dependencies.research_limits
        self._event_loop = asyncio.get_running_loop()
        self._deadline = run_deadline or time.monotonic() + self._limits.deadline_s
        self._sub_agent_gate = asyncio.Semaphore(self._limits.max_parallel_sub_agents)

        # Document text stays here, and only ids reach the plan.
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        self._delegated_chunk_ids: set[str] = set()
        self._analysed_chunk_ids: set[str] = set()
        self._search_calls = 0
        self._batches_by_depth: Counter[int] = Counter()
        self.sub_agent_calls = sub_agent_calls_used
        self.depth_reached = 0
        self.findings: list[Finding] = []
        self.degraded: list[str] = []

    def build_sandbox_names(self, question: str, plan: Plan) -> dict[str, Any]:
        """Return everything a plan may use: plain values and the helpers (rule 7)."""
        return {
            "question": question,
            "search_query": plan.search_query,
            "departments": list(plan.departments),
            "document_types": list(plan.document_types),
            "recent_days": plan.recent_days,
            "MAX_BATCHES": self._limits.max_batches_per_level,
            "explore_documents": self.explore_documents,
            "search": self.search,
            "group_by_doc": group_by_doc,
            "fetch_sections": self.fetch_sections,
            "chunk": chunk,
            "delegate": self.delegate,
            "days_ago": timestamp_days_ago,
        }

    @property
    def chunks_read(self) -> int:
        """Return how many chunks reached a sub-agent, for the panel's "read of total" line."""
        return len(self._analysed_chunk_ids)

    def evidence_chunks(self) -> list[RetrievedChunk]:
        """Return the chunks behind the findings. Rule 4: only these can be cited."""
        cited_chunk_ids: list[str] = []
        for finding in self.findings:
            for chunk_id in finding.evidence_chunk_ids:
                if chunk_id not in cited_chunk_ids:
                    cited_chunk_ids.append(chunk_id)
        return [self._chunks_by_id[chunk_id] for chunk_id in cited_chunk_ids]

    def explore_documents(
        self,
        document_types: Sequence[str] = (),
        departments: Sequence[str] = (),
        created_after: int | None = None,
    ) -> list[dict[str, Any]]:
        """List matching documents without their text. Rule 2: the catalog filters by access.

        The catalog is in memory, so this needs no event loop and costs no budget.
        """
        self._raise_if_past_deadline()
        entries = self._dependencies.document_catalog.list_documents(
            self._user,
            DocumentFilters(document_types=tuple(document_types), created_after_ts=created_after),
            tuple(departments) or None,
        )
        emit_event(EventType.RESEARCH, stage=ResearchStage.EXPLORED, document_count=len(entries))
        return [entry.model_dump(include=PLAN_VISIBLE_CATALOG_FIELDS) for entry in entries]

    def search(
        self,
        query: str,
        document_types: Sequence[str] = (),
        departments: Sequence[str] = (),
        created_after: int | None = None,
    ) -> list[dict[str, str]]:
        """Search without reranking, because a plan's searches would use up the monthly quota."""
        return self._run_on_event_loop(
            self._search(query, document_types, departments, created_after)
        )

    async def _search(
        self,
        query: str,
        document_types: Sequence[str],
        departments: Sequence[str],
        created_after: int | None,
    ) -> list[dict[str, str]]:
        """Run one budgeted search and keep its chunks for the sub-agents."""
        self._search_calls += 1
        if self._search_calls > self._limits.max_search_calls:
            raise ResearchBudgetExceeded(
                f"A plan may search at most {self._limits.max_search_calls} times."
            )

        # Rule 1: the user comes from the session, so the search applies the access filter.
        search_result = await self._dependencies.document_search.search(
            query,
            self._user,
            DocumentFilters(document_types=tuple(document_types), created_after_ts=created_after),
            namespaces=tuple(departments) or None,
            use_reranker=False,
            result_count=self._limits.search_result_count,
        )
        self._note_degraded(*search_result.degraded)
        self._remember_chunks(search_result.chunks)
        emit_event(
            EventType.RESEARCH,
            stage=ResearchStage.SEARCHED,
            query=query,
            result_count=len(search_result.chunks),
        )
        return [_describe_chunk(found_chunk) for found_chunk in search_result.chunks]

    def fetch_sections(
        self, doc_ids: Iterable[str], section_keyword: str | None = None
    ) -> list[dict[str, str]]:
        """Read targeted sections of named documents, rather than loading them whole."""
        return self._run_on_event_loop(self._fetch_sections(list(doc_ids), section_keyword))

    async def _fetch_sections(
        self, doc_ids: list[str], section_keyword: str | None
    ) -> list[dict[str, str]]:
        """Fetch the documents' chunks, and keep the sections the plan asked for."""
        fetched_chunks = await self._read_document_chunks(doc_ids)
        matching_chunks = fetched_chunks
        if section_keyword:
            lowered_keyword = section_keyword.lower()
            matching_chunks = [
                fetched_chunk
                for fetched_chunk in fetched_chunks
                if lowered_keyword in fetched_chunk.section.lower()
            ]
        emit_event(
            EventType.RESEARCH,
            stage=ResearchStage.SECTIONS_FETCHED,
            document_count=len(doc_ids),
            section_keyword=section_keyword,
            section_count=len(matching_chunks),
        )
        return [_describe_chunk(matching_chunk) for matching_chunk in matching_chunks]

    async def _read_document_chunks(self, doc_ids: list[str]) -> list[RetrievedChunk]:
        """Fetch every chunk of the named documents that this user may read.

        Rule 2: fetching by id bypasses the query filter, so access is checked twice. Ids come
        only from catalog entries the user may read, and returned metadata is checked again.
        """
        permitted_entries = self._dependencies.document_catalog.list_documents(self._user)
        requested_doc_ids = set(doc_ids)
        chunk_ids_by_namespace: dict[str, list[str]] = {}
        for entry in permitted_entries:
            if entry.doc_id in requested_doc_ids:
                chunk_ids_by_namespace.setdefault(entry.department, []).extend(entry.chunk_ids())
        if not chunk_ids_by_namespace:
            return []

        fetch_result = await self._dependencies.document_search.fetch_chunks(
            chunk_ids_by_namespace, self._user
        )
        self._note_degraded(*fetch_result.degraded)
        permitted_chunks = keep_permitted_chunks(fetch_result.chunks, self._user)
        self._remember_chunks(permitted_chunks)
        return permitted_chunks

    def delegate(self, task: str, batches: Sequence[Sequence[dict[str, Any]]]) -> None:
        """Send each batch to a sub-agent. Findings are recorded here, not returned to the plan."""
        self._run_on_event_loop(self._delegate(task, batches))

    async def _delegate(self, task: str, batches: Sequence[Sequence[dict[str, Any]]]) -> None:
        """Check the plan's request against the budget, and fan out to sub-agents."""
        chunk_batches = self._resolve_batches(batches)

        # A plan that asks for too much is a bad plan, so it fails rather than quietly shrinking.
        batches_requested = self._batches_by_depth[TOP_LEVEL] + len(chunk_batches)
        if batches_requested > self._limits.max_batches_per_level:
            raise ResearchBudgetExceeded(
                f"delegate accepts at most {self._limits.max_batches_per_level} batches, "
                f"but {batches_requested} were requested."
            )

        emit_event(
            EventType.RESEARCH,
            stage=ResearchStage.BATCHES_PLANNED,
            depth=TOP_LEVEL,
            batch_sizes=[len(chunks) for chunks in chunk_batches],
        )
        await asyncio.gather(
            *[self._analyse_batch(task, chunks, TOP_LEVEL) for chunks in chunk_batches]
        )

    def _resolve_batches(
        self, batches: Sequence[Sequence[dict[str, Any]]]
    ) -> list[list[RetrievedChunk]]:
        """Turn a plan's records into chunks, each sent to a sub-agent only once.

        One chunk can cover several headings, so fetching by two keywords returns it twice.
        Only chunks this session read can be resolved, so a made-up id reaches nothing.
        """
        chunk_batches = []
        for batch in batches:
            new_chunks = []
            for record in batch:
                known_chunk = self._chunks_by_id.get(record["chunk_id"])
                if known_chunk is None or known_chunk.chunk_id in self._delegated_chunk_ids:
                    continue
                self._delegated_chunk_ids.add(known_chunk.chunk_id)
                new_chunks.append(known_chunk)
            if new_chunks:
                chunk_batches.append(new_chunks)
        return chunk_batches

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
        """Read the other sections of documents the sub-agent could not resolve."""
        unresolved_doc_ids = sorted(
            {
                chunks[marker - 1].doc_id
                for marker in analysis.unresolved_markers
                if 1 <= marker <= len(chunks)
            }
        )
        if not unresolved_doc_ids or depth >= self._limits.max_depth:
            return

        already_read_ids = {batch_chunk.chunk_id for batch_chunk in chunks}
        document_chunks = await self._read_document_chunks(unresolved_doc_ids)
        unread_chunks = [
            document_chunk
            for document_chunk in document_chunks
            if document_chunk.chunk_id not in already_read_ids
        ]
        if unread_chunks:
            await self._analyse_batch(task, unread_chunks, depth + 1)

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

        # Rule 5: the sections reach the sub-agent as data, inside document tags.
        request_text = SUB_AGENT_REQUEST_TEMPLATE.format(
            task=task, documents=format_evidence_documents(chunks)
        )
        messages: list[BaseMessage] = [
            SystemMessage(SUB_AGENT_SYSTEM_PROMPT),
            HumanMessage(request_text),
        ]
        async with self._sub_agent_gate:
            try:
                analysis_result = await self._dependencies.llm.generate_structured(
                    SUB_AGENT_ROUTE, messages, BatchAnalysis
                )
            except Exception as error:  # One failed sub-agent must not fail the whole run.
                logger.warning("research_sub_agent_failed", error=type(error).__name__)
                self._note_degraded(DEGRADED_BATCH_FAILED)
                return None

        self._note_degraded(
            *[
                f"{DEGRADED_MODEL_FALLBACK}:{model_spec}"
                for model_spec in analysis_result.failed_model_specs
            ]
        )
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

    def _remember_chunks(self, chunks: Iterable[RetrievedChunk]) -> None:
        """Hold chunk text in the session, because a plan handles ids only."""
        for retrieved_chunk in chunks:
            self._chunks_by_id[retrieved_chunk.chunk_id] = retrieved_chunk

    def _note_degraded(self, *reasons: str) -> None:
        """Record what went wrong once, however many batches it affected."""
        for reason in reasons:
            if reason not in self.degraded:
                self.degraded.append(reason)

    def _build_deadline_error(self) -> ResearchBudgetExceeded:
        """Describe a run that is out of time, in the words the next plan attempt will read."""
        return ResearchBudgetExceeded(
            f"The research run passed its {self._limits.deadline_s} s deadline."
        )

    def _raise_if_past_deadline(self) -> None:
        """Stop a plan that has run out of time."""
        if time.monotonic() >= self._deadline:
            raise self._build_deadline_error()

    def _run_on_event_loop[Result](self, work: Coroutine[Any, Any, Result]) -> Result:
        """Run async work on the event loop from the sandbox thread, bounded by the deadline."""
        seconds_left = self._deadline - time.monotonic()
        if seconds_left <= 0:
            work.close()
            raise self._build_deadline_error()

        future = asyncio.run_coroutine_threadsafe(work, self._event_loop)
        try:
            return future.result(timeout=seconds_left)
        except TimeoutError as error:
            # Cancelling stops the sub-agents, and findings recorded so far are kept.
            future.cancel()
            raise self._build_deadline_error() from error
