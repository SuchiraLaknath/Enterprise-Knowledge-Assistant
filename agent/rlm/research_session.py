"""One research run, and the helpers its plan can call.

A plan is code written by a model, so what it can reach matters. `ResearchSession` holds the
verified user and the document text, and a plan only ever receives ids, titles, sections and
dates. No helper takes a user or an access level, so a plan cannot ask for more than the
employee may read (rules 1 and 7).

Helpers are synchronous, because plans run in the sandbox thread, but search and LLM calls are
async. `_run_on_event_loop` hops back to the event loop for that work. Waiting there uses none
of the sandbox's CPU budget.

Rule 7: a helper must read everything a plan passes it before that hop. A plan can pass a
generator, whose body is plan code, and on the event loop it would escape the CPU limit and
freeze every user's request. So only plain values built in the sandbox thread cross over.
"""

import asyncio
import time
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any

from agent.dependencies import AgentDependencies
from agent.rlm.sub_agents import ResearchBudgetExceeded, SubAgentTeam
from agent.schemas import Finding, Plan
from api.events import EventType, ResearchStage, emit_event
from retrieval.filters import keep_permitted_chunks, timestamp_days_ago
from retrieval.models import DocumentFilters, RetrievedChunk, UserContext

# What a plan may see of a catalog entry. It holds no text, and `access_level` is not needed.
PLAN_VISIBLE_CATALOG_FIELDS = {
    "doc_id",
    "title",
    "department",
    "document_type",
    "created_date",
    "chunk_count",
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
    """One attempt at a research run: its helpers, and everything it has read and found.

    Each helper a plan calls is a plain method, and the async work it needs has the same name
    with a leading underscore. Search and batch limits apply to one plan, and sub-agent calls and
    the deadline apply to the whole run, across plan attempts.
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

        # Document text stays here, and only ids reach the plan.
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        self._delegated_chunk_ids: set[str] = set()
        self._search_calls = 0
        self.degraded: list[str] = []
        self.sub_agents = SubAgentTeam(
            dependencies.llm,
            self._limits,
            self._read_document_chunks,
            self._note_degraded,
            sub_agent_calls_used,
        )

    def build_sandbox_names(self, question: str, plan: Plan) -> dict[str, Any]:
        """Return everything a plan may use: plain values and the helpers (rule 7)."""
        return {
            "question": question,
            "search_query": plan.search_query,
            "departments": list(plan.departments),
            "document_types": list(plan.document_types),
            "recent_days": plan.recent_days,
            "TARGET_BATCHES": self._limits.plan_batch_target,
            "explore_documents": self.explore_documents,
            "search": self.search,
            "group_by_doc": group_by_doc,
            "fetch_sections": self.fetch_sections,
            "chunk": chunk,
            "delegate": self.delegate,
            "days_ago": timestamp_days_ago,
        }

    @property
    def findings(self) -> list[Finding]:
        """Return what the sub-agents have found so far."""
        return self.sub_agents.findings

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
        # An empty query could fail at Pinecone, and failures trip a breaker shared by all users.
        if not str(query).strip():
            raise ValueError("A search needs a query.")
        return self._run_on_event_loop(
            self._search(str(query), tuple(document_types), tuple(departments), created_after)
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

    def delegate(self, task: str, batches: Iterable[Iterable[dict[str, Any]]]) -> None:
        """Send each batch to a sub-agent. Findings are recorded here, not returned to the plan."""
        chunk_id_batches = []
        for batch in batches:
            chunk_id_batches.append([str(record["chunk_id"]) for record in batch])
        self._run_on_event_loop(self._delegate(str(task), chunk_id_batches))

    async def _delegate(self, task: str, chunk_id_batches: list[list[str]]) -> None:
        """Turn the plan's ids back into chunks, and hand them to the sub-agents."""
        chunk_batches = self._rebalance_batches(self._resolve_batches(chunk_id_batches))
        await self.sub_agents.analyse_batches(task, chunk_batches)

    def _rebalance_batches(
        self, chunk_batches: list[list[RetrievedChunk]]
    ) -> list[list[RetrievedChunk]]:
        """Even out batches too large to split once, so that no section is left unread.

        Removing duplicates can leave a plan's batches uneven, and a batch over twice one reader's
        limit would be cut short at the deepest level. Sorting by chunk id keeps each document's
        sections together.
        """
        largest_splittable_batch = self._limits.sub_agent_chunk_limit * 2
        if all(len(chunks) <= largest_splittable_batch for chunks in chunk_batches):
            return chunk_batches

        all_chunks = sorted(
            (retrieved_chunk for chunks in chunk_batches for retrieved_chunk in chunks),
            key=lambda retrieved_chunk: retrieved_chunk.chunk_id,
        )
        target_batches = self._limits.plan_batch_target
        batch_size = (len(all_chunks) + target_batches - 1) // target_batches
        return chunk(all_chunks, batch_size)

    def _resolve_batches(self, chunk_id_batches: list[list[str]]) -> list[list[RetrievedChunk]]:
        """Turn a plan's chunk ids into chunks, each sent to a sub-agent only once.

        One chunk can cover several headings, so fetching by two keywords returns it twice.
        Only chunks this session read can be resolved, so a made-up id reaches nothing.
        """
        chunk_batches = []
        for chunk_ids in chunk_id_batches:
            new_chunks = []
            for chunk_id in chunk_ids:
                known_chunk = self._chunks_by_id.get(chunk_id)
                if known_chunk is None or known_chunk.chunk_id in self._delegated_chunk_ids:
                    continue
                self._delegated_chunk_ids.add(known_chunk.chunk_id)
                new_chunks.append(known_chunk)
            if new_chunks:
                chunk_batches.append(new_chunks)
        return chunk_batches

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
