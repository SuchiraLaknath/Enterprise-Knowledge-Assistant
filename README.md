# Enterprise Knowledge Assistant

A multi-agent assistant for Commercial Bank employees, built for the AI Lead technical assessment. It answers questions from internal documents with cited evidence, shows its reasoning live, and respects each employee's role. All data is mock data.

| What you might look for | Where |
|---|---|
| Architecture and diagrams | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Assumptions and trade-offs | [docs/adr/](docs/adr/) |

## What it does

- **Answers with citations.** Hybrid search over Pinecone, with reranking. A citation that points at nothing is rejected.
- **Investigates whole collections.** A planner writes a Python plan, a sandbox runs it, and sub-agents read targeted sections, so large questions never load whole documents into the model.
- **Shows its work.** A live activity panel shows each agent step, tool call, memory update and validation, and LangSmith holds the full trace.
- **Respects roles.** Access is enforced in code at three points, and the LLM never decides it.
- **Resists attack.** Layered defences cover direct and indirect prompt injection, and write tools wait for a person.
- **Degrades gracefully.** Each failure has a defined fallback, and the employee sees which one was used.

## Run it

You need Docker, and API keys for Anthropic, Pinecone and LangSmith. OpenAI and Groq keys are optional, and each adds a fallback provider ([ADR 0004](docs/adr/0004-model-selection.md)).

```bash
cp .env.example .env     # then fill in the keys. Generate JWT_SECRET: openssl rand -hex 32
docker compose --profile ingest run --rm ingest    # first time only: loads the mock corpus
docker compose up --build
```

Open <http://localhost:8501>, and choose a demo user:

| User | Role | Documents | Tools |
|---|---|---|---|
| `vimal` | viewer | public, internal | chat, `knowledge_search` |
| `nadee` | analyst | + confidential | + deep research, MCP tools, Python analysis |
| `admin` | admin | + restricted | + audit log, incident updates with approval |

Try these as `nadee`:

- How many days of annual leave can I carry over?
- Summarize all outage reports related to payment failures during the last year and identify recurring root causes.
- Who owns the Card Switch service, and which incidents are open right now?

Sign-in is one click for the demo, so ports bind to this machine only ([ADR 0010](docs/adr/0010-demo-sign-in.md)).

### Run it without containers

For Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
docker compose up -d redis postgres
uv run python -m nltk.downloader punkt_tab stopwords    # once per machine
uv run python -m mcp_server.server                      # terminal 1
uv run uvicorn api.main:app --reload                    # terminal 2
uv run streamlit run ui/streamlit_app.py                # terminal 3
```

## Test it

```bash
uv run pytest                       # all tests, which use fakes and need no network or keys
uv run pytest -m faults             # failure injection
uv run ruff check . && uv run ruff format --check .
uv run mypy config ingestion resilience retrieval agent api security tools evals
uv run python -m evals.run_retrieval_eval    # live retrieval quality, using no rerank quota
```

To see the fallbacks live, set `FAULTS` before starting the API. It accepts `pinecone`, `rerank`, `mcp`, `anthropic`, `openai`, `groq`, `llm`, `tool_timeout`, `redis` and `postgres`, separated by commas.

## Results

Checked live against the seeded root causes in the mock corpus, on the demo question:

| User | Documents found | Themes |
|---|---|---|
| `nadee` | 22 of 22 | Connection pools 8, certificates 7, gateway 5, batch jobs 2 |
| `admin` | 24 of 24 | Connection pools 8, certificates 8, gateway 6, batch jobs 2 |

A run takes 30 to 50 seconds. In about a third of runs the validator rejects the first draft, usually for a claim that goes beyond the evidence, and the rewritten answer passes. It reads only the section holding each root cause, so 22 of the 154 chunks an analyst may read. Retrieval scores a 100% hit rate on the golden questions in `evals/`, which is a small set.

Public LangSmith traces, from runs on the primary models, show every step of a turn. Document text is hidden, as [ADR 0008](docs/adr/0008-tracing-redaction.md) explains:

- [The demo question](https://smith.langchain.com/public/bf1a3f60-e3e2-441c-8ab4-92a81cb14ace/r/0172920c-a361-473d-9f3b-1de9320b426e?start_time=2026-10-31T15%3A12%3A00.063816Z) as `nadee`: 22 of 22 documents.
- [A whole-report question](https://smith.langchain.com/public/05c59fee-4217-483a-8a23-148ccd1e6cdc/r/e2b1a24f-43d3-4be9-b211-075122038659?start_time=2026-10-31T15%3A21%3A18.384698Z) as `admin`: 24 of 24 documents, reaching recursion depth 2.

## How it is built

| Layer | Choice |
|---|---|
| UI | Streamlit |
| API | FastAPI, async, with server-sent events |
| Orchestration | LangGraph: supervisor pattern, with research as a subgraph |
| LLM | An in-process gateway over Claude, with OpenAI and Groq as fallbacks |
| Search | Pinecone hybrid (dense and BM25), namespaces per department, reranking |
| Tools | A role-checked registry, FastMCP server, sandboxed Python |
| State | Postgres for checkpoints, memory, audit and feedback. Redis for rate limits |
| Observability | LangSmith, and structured JSON logs |

```text
agent/        graph, nodes, prompts, the RLM research flow, and the LLM gateway
api/          FastAPI routes, auth, rate limiting, tracing
retrieval/    hybrid search, reranking, filters, the catalog and the local fallback
security/     the sandbox, guards and policies
tools/        the registry, tools and audit log
mcp_server/   dummy enterprise data over MCP
ingestion/    the mock corpus generator and indexer
ui/           Streamlit
evals/        retrieval scoring and golden questions
docs/         architecture and decisions
```

## Assumptions to know about

- **Dates move.** The corpus was generated for 2026-10-30, so running much later moves older incidents out of "the last year". Rerun `uv run python -m ingestion.generate_docs` and ingest again to refresh it, after changing its reference date.
- **Rerank quota.** Pinecone's Starter plan allows 500 reranks a month, so only the retrieval agent reranks.
- **Not production ready.** The sandbox runs inside the API process ([ADR 0001](docs/adr/0001-sandbox-limits.md)), sign-in is for demos, and retrieved text goes to fallback vendors, which real data would need approval for.
