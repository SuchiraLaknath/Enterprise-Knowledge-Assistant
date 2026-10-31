# 0004. Which models do which work

Date: 2026-10-31
Status: Accepted

## Context

The brief accepts any modern LLM, and asks for the choice to be justified. One turn makes several calls of very different weight: a plan needs judgement, while a guard or a sub-agent reading a few sections needs speed, and research can make up to 24 of those.

## Decision

Work is split into two routes in `config/models.yaml`, and no node names a model.

- **`response`:** Claude Sonnet, then Haiku, then OpenAI `gpt-5.5`, then Groq `openai/gpt-oss-120b`. It serves the supervisor, research planner, aggregator and answer, where quality shows directly.
- **`fast`:** Claude Haiku, then Groq `openai/gpt-oss-20b`, then OpenAI `gpt-5.4-mini`. It serves the input and answer judges, tool selection, sub-agents and summaries, where volume and latency matter most.

Haiku follows Sonnet because an overload on one often clears on the other. A second provider covers a full outage. Each model was checked on every structured output, on tool calls and on streaming before joining a route. Groq comes before OpenAI on `fast`, because small steps benefit most from its speed.

## Consequences

- Research stays affordable, because the many small steps use the cheap route.
- Changing a model is a one-line config change.
- This was proven when Anthropic credit ran out during testing, and the app carried on through OpenAI.
- Results vary by provider. Claude and OpenAI found every document on the demo question, though OpenAI sometimes splits a theme in two.
- Groq is a last resort. Its free tier allows 8,000 tokens a minute, and a research run needs more, so on Groq alone batches fail and the run reads about half the documents. A fixed notice then states the real coverage. In an Anthropic outage this does not arise, because OpenAI answers whatever Groq cannot.
