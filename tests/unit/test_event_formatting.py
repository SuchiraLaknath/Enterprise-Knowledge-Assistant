from ui.event_formatting import apply_token_event, describe_event, parse_sse_line


def test_node_events_show_the_active_node():
    assert describe_event({"type": "node", "node": "supervisor", "status": "started"}) == (
        "▶ supervisor"
    )
    assert describe_event({"type": "node", "node": "supervisor", "status": "finished"}) == (
        "✓ supervisor"
    )


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
