# 0001. What the Python sandbox stops, and what it does not

Date: 2026-10-31
Status: Accepted

## Context

RLM plans are Python written by a model, and a prompt injection could shape that code. CPython has no safe in-process sandbox, so anything built here is a set of layers rather than a guarantee. The sandbox must still be small enough for an evaluator to read in one sitting.

## Decision

`security/sandbox.py` uses two layers:

- **Before running:** an allow-list of syntax, names and attributes. Attributes are allow-listed, not just dunders, because `"{0.__class__}".format(value)` reaches interpreter objects through a string. A failed check means no line of the code runs.
- **While running:** a dedicated thread with a line trace that stops the code after 2 s of `time.thread_time()`, so that waiting on a search does not count.

`Mult`, `Pow`, `Mod` and format specifications are refused, and `range` is capped, because these can build a huge object in one C call. The plan also never receives document text, so generated code has nothing to leak.

The main alternative was a separate process or container per plan. It would give real memory and CPU limits, but it adds a service, and helpers would need an RPC layer to reach search and the verified user.

## Consequences

- **Not stopped:** a long operation in C cannot be interrupted partway, because the trace runs between Python lines. Repeated string doubling (`text = text + text` in a loop) can exhaust memory before the CPU limit fires, because each step takes little CPU.
- **Not limited:** memory has no limit of its own, and a plan runs inside the API process.
- **Accepted because:** this is a proof of concept, with mock data, on a machine the demo controls. Plans can only come from a signed-in analyst or admin, and the input guard in build step 8 screens the question first.
- **For production:** run plans in a separate process with memory limits, and keep the same allow-list as the first layer.
