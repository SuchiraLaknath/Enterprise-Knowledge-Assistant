# 0003. Deep research is for analysts and admins

Date: 2026-10-31
Status: Accepted

## Context

The brief's roles give viewers chat and search, analysts search, analytics and MCP tools, and administrators everything. It does not name research directly. One research run costs up to 24 sub-agent calls, where a normal answer costs two LLM calls, so opening it to every employee is a cost and rate-limit decision as well as an access one.

## Decision

`deep_research` is registered in `tools/registry.py` with a `min_role` of `analyst`. The plan may ask for research, but the verified role decides, in two places: `supervisor` before routing, and `check_research_access` at the entry of the subgraph (rule 2). A viewer's broad question gets normal retrieval, and the answer ends with a fixed note explaining the limit.

The alternative was to let viewers run research over the documents they can already read. The data risk is the same, because helpers apply the viewer's access. It was rejected on cost, and because analytics belongs to the analyst role in the brief.

## Consequences

- A viewer always gets an answer and knows why it may be incomplete, instead of a refusal.
- The note is added by code, so a model cannot leave it out or reword it.
- Giving viewers research later needs a one-word change to `min_role`, and it follows the same rules as every other tool.
- Routing starts research, so no model is ever offered it as a tool.
