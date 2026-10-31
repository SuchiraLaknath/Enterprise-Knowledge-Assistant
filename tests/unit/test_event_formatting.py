import pytest

from api.events import ResearchStage
from retrieval.models import DegradedReason
from ui.event_formatting import (
    DEGRADED_SENTENCES,
    RESEARCH_STAGE_DESCRIPTIONS,
    TurnStatus,
    add_activity_event,
    apply_token_event,
    describe_degraded,
    describe_event,
    describe_turn_status,
    describe_waiting_step,
    parse_sse_line,
    update_turn_status,
)


def test_node_events_show_the_active_node():
    assert describe_event({"type": "node", "node": "supervisor", "status": "started"}) == (
        "▶ supervisor: planning the work"
    )
    assert describe_event({"type": "node", "node": "supervisor", "status": "finished"}) == (
        "✓ supervisor: planning the work"
    )


def test_a_finished_node_ticks_its_own_line_and_keeps_its_details_below():
    lines = []
    add_activity_event(lines, {"type": "node", "node": "validator", "status": "started"})
    add_activity_event(
        lines, {"type": "validation", "check": "validator", "status": "pass", "reasons": []}
    )
    add_activity_event(lines, {"type": "node", "node": "validator", "status": "finished"})

    assert lines == ["✓ validator: checking the answer", "  Check validator: pass"]


def test_a_node_that_runs_twice_ticks_its_latest_line():
    # The supervisor runs again after each worker, so an earlier tick must not be disturbed.
    lines = []
    for status in ("started", "finished", "started"):
        add_activity_event(lines, {"type": "node", "node": "supervisor", "status": status})
    add_activity_event(lines, {"type": "node", "node": "supervisor", "status": "finished"})

    assert lines == ["✓ supervisor: planning the work", "✓ supervisor: planning the work"]


def test_a_finished_node_with_no_started_line_is_still_shown():
    # Nothing may vanish from the panel, even if an event arrives out of order.
    lines = []
    add_activity_event(lines, {"type": "node", "node": "new_node", "status": "finished"})

    assert lines == ["✓ new_node"]


def test_the_status_line_follows_the_turn():
    status = TurnStatus()
    assert describe_turn_status(status) == "Waiting for a question."

    update_turn_status(status, {"type": "node", "node": "retrieval_agent", "status": "started"})
    update_turn_status(status, {"type": "state", "evidence_count": 8})
    update_turn_status(status, {"type": "state", "response_retries": 1})
    assert describe_turn_status(status) == "Now: retrieval_agent | Evidence: 8 | Retries: 1"
    assert describe_waiting_step(status).startswith("Searching the documents.")

    update_turn_status(status, {"type": "final"})
    assert describe_turn_status(status) == "Finished | Evidence: 8 | Retries: 1"


def test_degraded_ids_become_plain_sentences_once_each():
    sentences = describe_degraded(
        [
            "input_guard_used_fallback_model:anthropic:claude-haiku-5-5",
            "supervisor_used_fallback_model:anthropic:claude-sonnet-5-5",
            "pinecone_unavailable",
            "local_keyword_search",
        ]
    )

    assert sentences == [
        "A backup AI model answered some steps.",
        "Document search was unavailable, so a local keyword search was used.",
    ]


def test_an_unknown_degraded_id_is_shown_rather_than_hidden():
    assert describe_degraded(["something_new_failed"]) == ["something_new_failed"]


@pytest.mark.parametrize("reason", list(DegradedReason))
def test_every_retrieval_failure_has_a_plain_sentence(reason):
    assert reason.value in DEGRADED_SENTENCES


def test_state_events_show_the_plan_and_any_degradation():
    description = describe_event(
        {
            "type": "state",
            "intent": "Find the cause",
            "plan_steps": ["Search"],
            "evidence_count": 3,
            "degraded": ["rerank_unavailable"],
        }
    )

    assert "Find the cause" in description
    assert "Step 1: Search" in description
    assert "3 chunks" in description
    assert "rerank_unavailable" in description


def test_route_events_show_the_reason():
    description = describe_event(
        {"type": "route", "source": "supervisor", "target": "response_agent", "reason": "ready"}
    )

    assert description == "  Route: supervisor → response_agent (ready)"


def test_retrieval_events_show_the_query_and_sources():
    description = describe_event(
        {
            "type": "retrieval",
            "query": "expired certificate",
            "departments": "all",
            "dense_weight": 0.6,
            "result_count": 1,
            "sources": [{"score": 0.91234, "chunk_id": "DOC#c0", "title": "Title"}],
        }
    )

    assert "expired certificate" in description
    assert "0.912 DOC#c0 Title" in description


def test_chat_events_are_left_out_of_the_panel():
    assert describe_event({"type": "token", "text": "Hello"}) is None
    assert describe_event({"type": "final", "answer": "Hello"}) is None


def test_tokens_build_the_draft_and_a_reset_clears_it():
    draft = apply_token_event("", {"type": "token", "text": "Partial "})
    draft = apply_token_event(draft, {"type": "token", "text": "answer"})
    assert draft == "Partial answer"

    assert apply_token_event(draft, {"type": "token", "reset": True}) == ""


def test_only_data_lines_are_read_from_the_stream():
    assert parse_sse_line('data: {"type": "node"}') == '{"type": "node"}'
    assert parse_sse_line(": ping - 2026-10-31") is None
    assert parse_sse_line("") is None


def test_a_research_plan_is_shown_in_full():
    description = describe_event(
        {
            "type": "research",
            "stage": "plan_ready",
            "source": "llm",
            "attempt": 1,
            "code": "documents = explore_documents()\ndelegate(question, [])",
        }
    )

    assert "attempt 1, from llm" in description
    assert "documents = explore_documents()" in description
    assert "delegate(question, [])" in description


def test_a_research_report_shows_depth_and_chunks_read_against_the_total():
    description = describe_event(
        {
            "type": "research",
            "stage": "report_ready",
            "themes": [{"name": "Certificate expiry", "document_count": 8}],
            "documents_analysed": 22,
            "documents_in_scope": 24,
            "chunks_read": 24,
            "chunks_available": 150,
            "depth_reached": 2,
            "sub_agent_calls": 9,
        }
    )

    assert "22 of 24 documents analysed" in description
    assert "24 of 150 chunks read" in description
    assert "Depth reached: 2, sub-agents: 9" in description
    assert "8 documents: Certificate expiry" in description


def test_each_research_stage_has_a_description():
    stages = [
        {"stage": "explored", "document_count": 22},
        {"stage": "searched", "query": "outage", "result_count": 5},
        {
            "stage": "sections_fetched",
            "section_keyword": None,
            "section_count": 22,
            "document_count": 22,
        },
        {"stage": "batches_planned", "depth": 1, "batch_sizes": [4, 4]},
        {
            "stage": "batch_finished",
            "depth": 2,
            "finding_count": 3,
            "batch_size": 4,
            "chunks_read": 8,
        },
        {
            "stage": "plan_failed",
            "error": "SandboxViolation: Import is not allowed.",
            "findings_kept": 0,
        },
    ]

    descriptions = [describe_event({"type": "research", **stage}) for stage in stages]

    assert "22 documents" in descriptions[0]
    assert "outage" in descriptions[1]
    assert "all sections" in descriptions[2]
    assert "[4, 4]" in descriptions[3]
    assert "depth 2" in descriptions[4]
    assert "Import is not allowed" in descriptions[5]


def test_every_research_stage_has_a_description():
    # The UI names stages as plain strings, so this stops the two lists from drifting apart.
    assert set(RESEARCH_STAGE_DESCRIPTIONS) == {stage.value for stage in ResearchStage}


def test_an_unknown_stage_is_never_reported_as_a_failure():
    assert describe_event({"type": "research", "stage": "new_stage"}) == "  Research: new_stage"


def test_a_plan_that_stopped_early_says_that_its_findings_were_kept():
    description = describe_event(
        {"type": "research", "stage": "plan_failed", "error": "IndexError", "findings_kept": 4}
    )

    assert "keeping 4 findings" in description


def test_tool_events_show_the_call_and_its_outcome():
    description = describe_event(
        {
            "type": "tool",
            "tool_name": "mcp.service_catalog",
            "arguments": {"service_name": "Card Switch"},
            "status": "ok",
        }
    )

    assert "mcp.service_catalog" in description
    assert "Card Switch" in description
    assert "ok" in description


def test_memory_events_show_the_operation_and_tier():
    description = describe_event(
        {"type": "memory", "operation": "write", "tier": "summary", "item_count": 6}
    )

    assert description == "  Memory write (summary): 6 items"


def test_validation_events_show_the_status_and_reasons():
    description = describe_event(
        {"type": "validation", "check": "validator", "status": "fail", "reasons": ["Dead [9]."]}
    )

    assert description == "  Check validator: fail: Dead [9]."


def test_approval_events_show_what_is_waiting():
    description = describe_event(
        {"type": "approval", "tool_name": "admin.write", "arguments": {"id": 1}}
    )

    assert "Waiting for approval: admin.write" in description
