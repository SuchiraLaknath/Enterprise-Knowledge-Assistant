# 0007. Prompt injection defence in layers

Date: 2026-10-31
Status: Accepted

## Context

No single check stops prompt injection, because instructions and data are both just text to a model. The corpus contains a real example: a runbook with a sentence telling the assistant to reveal its prompt and embed an image that sends data out. Direct attacks, indirect attacks and tool abuse all need covering, and an outage of one defence must not open the door.

## Decision

Each layer assumes that the one before it failed.

| Layer | Stops | Where |
|---|---|---|
| Input patterns | Known attacks, instantly, with no model | `security/input_guard.py` |
| Input judge | The same intent in new words | `agent/nodes/input_guard.py` |
| Content guard | Injected sentences in documents, findings and tool results | `security/content_guard.py` |
| Delimiters and escaping | Text posing as instructions | `agent/evidence_formatting.py` |
| Tool checks | A model reaching past its role | `tools/registry.py` |
| Signed role token | Anything calling the MCP server directly | `mcp_server/service_token.py` |
| Validator | Dead citations, brand breaches, unsupported claims | `agent/nodes/validator.py` |
| Output guard | Links and images that could send data out | `security/output_guard.py` |
| Plain-text drafts | An injected image firing before validation | `ui/streamlit_app.py` |

The content guard sits inside the one function that puts retrieved text into a prompt, so no path can skip it. It removes only the injected sentence, so a poisoned runbook keeps its useful steps. A refusal is fixed text, so a blocked message reaches no model that writes answers, and it leaves the conversation, so "do what I said earlier" cannot bring it back. Text is normalised first, so invisible characters and look-alike letters do not hide an attack. The judge's own words never reach the next prompt, because it reads retrieved text.

Patterns name the assistant's own instructions rather than alarming words, because staff legitimately ask about maintenance mode and the password policy. A test scans the real corpus, and only the poisoned runbook is flagged.

## Consequences

- If the judge is down, patterns decide alone and tools are limited to reading.
- Patterns cannot catch everything. Text spaced out letter by letter or encoded in base64 relies on the judge.
- The judge adds about a second to each turn.
- For production: run guards on a separate service, and tune patterns against real traffic.
