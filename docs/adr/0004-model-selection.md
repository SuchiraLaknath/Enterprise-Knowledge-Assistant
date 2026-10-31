# 0004. Which models do which work

Date: 2026-10-31
Status: Accepted

## Context

The brief accepts any modern LLM, and asks for the choice to be justified. One turn makes several calls of very different weight: a plan needs judgement, while a guard or a sub-agent reading a few sections needs speed, and research can make up to 24 of those.

## Decision

Work is split into two routes in `config/models.yaml`, and no node names a model.

- **`response`:** Claude Sonnet, then Haiku, then OpenAI `gpt-5.5`. It serves the supervisor, research planner, aggregator and answer, where quality shows directly.
- **`fast`:** Claude Haiku, then OpenAI `gpt-5.4-mini`. It serves the input and answer judges, tool selection, sub-agents and summaries, where volume and latency matter most.

Haiku follows Sonnet because an overload on one often clears on the other. A second provider covers a full outage. Each model was checked on every structured output and on streaming before joining a route. Groq is supported, but has no model on a route, because an untested model must not join one.

## Consequences

- Research stays affordable, because the many small steps use the cheap route.
- Changing a model is a one-line config change.
- This was proven when Anthropic credit ran out during testing, and the app carried on through OpenAI.
- Results vary a little by provider. Live runs on both found every document, but OpenAI sometimes splits a theme in two.
