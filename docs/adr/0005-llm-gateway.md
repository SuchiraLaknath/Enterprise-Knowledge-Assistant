# 0005. An in-process LLM gateway, not LiteLLM

Date: 2026-10-31
Status: Accepted

## Context

The system needs fallback between models and providers, timeouts, circuit breakers and visible failures. LiteLLM is the common choice. But malicious releases of it reached PyPI twice in 2026, and one payload stole environment variables, which is where this project keeps its keys.

## Decision

`agent/llm_gateway.py` is the only module that creates LLM clients. It walks a route's chain in a plain loop, with a timeout per attempt, a deadline per route and a circuit breaker per provider. `with_fallbacks` was rejected too, because a plain loop can report each attempt to `state["degraded"]`, the UI and LangSmith.

## Consequences

- About 280 readable lines, with no proxy to run and fewer dependencies with access to the keys.
- Every failed attempt is visible to the employee and in the trace.
- If a model fails partway through streaming, the draft is cleared and the next model starts clean.
- It covers only what this project needs. It has no cost dashboard or virtual keys, which a production platform might want from a dedicated gateway.
