# 0006. Three tiers of memory, and what never enters them

Date: 2026-10-31
Status: Accepted

## Context

The brief asks for memory across turns, and for the design to be explained. Long-term memory is also a risk: anything stored outlives the conversation, so an injected instruction stored once would act forever.

## Decision

| Tier | Holds | Storage |
|---|---|---|
| Thread | This conversation's messages | `AsyncPostgresSaver`, by thread |
| Summary | A rolling summary of older turns | Thread state |
| Long-term | Facts the employee stated, and doc ids | `AsyncPostgresStore`, by `user_id` |

Five rules keep the long-term tier safe:

- Facts are extracted only from what the employee typed, never from documents or answers, so a poisoned document cannot become a memory.
- Every fact passes the injection checks, and is refused if it holds a NIC or card number.
- Doc ids are stored, never document text.
- The namespace is the verified `user_id`, so nobody can read another person's memory.
- Answers join the conversation only after the validator passes them, and a blocked message is removed.

Facts are read by recency rather than by semantic search. A person has a handful of facts, so reading them all is simpler than adding an embedding call to every turn.

## Consequences

- Memory is a bonus, so any failure switches it off for that turn and the answer still goes out.
- With no first-person words in a message, extraction is skipped, which saves an LLM call on most turns.
- A fact can go stale, and there is no way yet for an employee to view or delete theirs. Production would need both, for privacy.
