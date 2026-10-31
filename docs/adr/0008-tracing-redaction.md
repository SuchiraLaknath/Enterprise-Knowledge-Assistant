# 0008. What reaches LangSmith

Date: 2026-10-31
Status: Accepted

## Context

The brief makes LangSmith mandatory, and evaluators must be able to inspect traces. But LangSmith is an external service, and LangChain traces every node input and output, which include documents and prompts.

## Decision

`api/tracing.py` redacts every payload in one place, through LangSmith's `anonymizer`, the only hook that also covers a run's `error` field.

| Hidden | Kept |
|---|---|
| Document bodies and chunk text | Chunk ids, titles, sections, scores |
| Research claims and theme summaries | Finding ids, labels, evidence ids |
| Tool data and tool arguments | Tool names, statuses |
| Memory and conversation summaries | That memory was read or written |
| Excerpt answers, which are raw document text | System prompts, the question |
| Error text, which can quote a whole reply | The error type |

**Theme names and final answers stay visible.** They sit close to document content, but hiding them would leave a trace that cannot show what the assistant said, which the brief needs evaluators to see. The project owner made this decision, and all data here is mock data.

## Consequences

- A node cannot forget to redact, because redaction happens as data leaves the process.
- Debugging from a trace alone can need a local rerun, as the text is not there.
- For production: real documents would need the answers hidden too, or a self-hosted LangSmith.
