# 0002. RLM research is built on LangGraph, not the Deep Agents library

Date: 2026-10-31
Status: Accepted

## Context

The brief mentions Deep Agents in the activity panel, and asks for recursive exploration with generated Python plans. A library could supply planning, sub-agents and a file-based workspace. But the rules here need control over exactly what generated code can reach, and every step must be visible in the panel and in LangSmith.

## Decision

Research is a LangGraph subgraph, `research_agent`, with plain Python helpers in `agent/rlm/`. It follows the Recursive Language Model pattern directly: the planner writes Python, a sandbox runs it, and helpers delegate batches to sub-agents, down to depth 2.

The Deep Agents library was rejected because its power comes from giving a model general tools. Here the opposite is wanted: seven narrow helpers, each of which applies the access rules itself, with typed records rather than free text between agents.

## Consequences

- Access checks, budgets and the fallback ladder are in this repository, where an evaluator can read and test them.
- Each step appears as a graph node, and the ladder reads as graph edges: LLM plan, LLM plan with the error, template plan, then plain retrieval.
- The cost is more code to own, and no ready-made planning or memory features.
- Plans run once and cannot see their results. A model that could look at results and write a second plan would be more flexible, but it would need document text or summaries to reach the planner, which is what keeps a poisoned document away from the code.
