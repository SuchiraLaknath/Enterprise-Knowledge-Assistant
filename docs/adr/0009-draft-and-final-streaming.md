# 0009. Streaming a draft, and then a final answer

Date: 2026-10-31
Status: Accepted

## Context

The brief asks for streaming responses, but the validator can only judge a finished answer. Streaming straight to the screen would show unvalidated text, and if it were rendered as markdown, an injected image would load before any check ran.

## Decision

Tokens stream live as a provisional draft, and only the `final` event is authoritative.

- The draft passes through a masker that holds back partial NIC and card numbers split across tokens.
- The draft is rendered as plain text, so nothing in it can load or link.
- Only the validated answer is rendered as markdown, and only it joins the conversation.
- `final` says whether it confirms or replaces the draft, and a retry clears the draft first.
- The trace link follows `final`, so a slow lookup can never delay an answer.

The alternative was to buffer until validated, which is simpler and safer, but the first word would take many seconds longer.

## Consequences

- Employees see progress at once, and what they keep is always validated.
- An employee can briefly read a draft that is later replaced. It is plain text and masked, and the screen shows the replacement.
- The rejected text remains in the turn's checkpoints, though never in the conversation.
