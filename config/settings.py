"""Typed access to configuration, so that a bad value fails at startup with a clear message.

Secrets come from the environment, and tunable values come from `config/*.yaml`. Each
build step adds the settings it needs here.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIRECTORY = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIRECTORY.parent

# The Markdown corpus, written by `ingestion/generate_docs.py`.
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "docs"
# BM25 word statistics, written by `ingestion/index.py` and read when encoding queries.
BM25_PARAMETERS_PATH = PROJECT_ROOT / "data" / "bm25_params.json"
# One summary per document, written by `ingestion/index.py` for the research planner.
DOCUMENT_CATALOG_PATH = PROJECT_ROOT / "data" / "doc_catalog.json"


class EnvironmentSettings(BaseSettings):
    """Values that differ per machine or are secret, loaded from the environment or `.env`."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Empty by default, because the UI and MCP containers receive no keys. The API refuses to
    # start without these, and ingestion cannot run without them.
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index: str = ""
    # The gateway skips a provider with no key, and refuses to start without Anthropic.
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_tracing: str = "false"
    langsmith_project: str = "enterprise-knowledge-assistant"
    # Empty by default, so that unit tests need no secret. The API refuses to start without it.
    jwt_secret: SecretStr = SecretStr("")
    database_url: str = ""
    redis_url: str = ""
    mcp_url: str = ""
    api_url: str = "http://localhost:8000"
    # Comma-separated names of clients to fail on purpose, for the degradation demo.
    faults: str = ""

    @property
    def enabled_faults(self) -> set[str]:
        """Return the fault names, so callers can test membership instead of parsing text."""
        return {fault.strip() for fault in self.faults.split(",") if fault.strip()}


def export_tracing_settings(settings: EnvironmentSettings) -> None:
    """Copy LangSmith settings into the process environment.

    Pydantic reads `.env` without exporting it, but LangSmith reads only the environment, so
    without this step tracing would be silently off.
    """
    os.environ["LANGSMITH_TRACING"] = settings.langsmith_tracing
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()


class ChunkingLimits(BaseModel):
    """Size rules for splitting documents into chunks."""

    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_fraction: float = Field(ge=0, lt=1)
    tokens_per_word: float = Field(gt=0)


class IndexingLimits(BaseModel):
    """Batch sizes and location for writing to Pinecone."""

    # Pinecone Inference accepts at most 96 inputs per call.
    embedding_batch_size: int = Field(gt=0, le=96)
    upsert_batch_size: int = Field(gt=0)
    cloud: str
    region: str


class RetrievalLimits(BaseModel):
    """Namespaces, score weights, result counts and timeouts for search."""

    namespaces: tuple[str, ...] = Field(min_length=1)
    default_dense_weight: float = Field(ge=0, le=1)
    keyword_query_dense_weight: float = Field(ge=0, le=1)
    candidates_per_namespace: int = Field(gt=0)
    rerank_candidate_count: int = Field(gt=0)
    minimum_rerank_score: float = Field(ge=0, le=1)
    final_result_count: int = Field(gt=0)
    pinecone_timeout_s: float = Field(gt=0)
    rerank_timeout_s: float = Field(gt=0)


class CircuitBreakerLimits(BaseModel):
    """When to stop calling a failing service, and when to try it again."""

    failure_threshold: int = Field(gt=0)
    recovery_seconds: float = Field(gt=0)


class LlmLimits(BaseModel):
    """Bounds on LLM calls, so that a failing model cannot hold a request open."""

    attempt_timeout_s: float = Field(gt=0)
    route_deadline_s: float = Field(gt=0)
    max_retries_per_model: int = Field(ge=0)
    structured_max_tokens: int = Field(gt=0)
    answer_max_tokens: int = Field(gt=0)


class ConversationLimits(BaseModel):
    """How much of a conversation is sent back to the model, and when older turns are condensed."""

    history_message_count: int = Field(gt=0)
    summarize_after_messages: int = Field(gt=0)
    max_memories_read: int = Field(gt=0)
    max_facts_per_turn: int = Field(gt=0)
    max_fact_characters: int = Field(gt=0)
    memory_timeout_s: float = Field(gt=0)


class ApiLimits(BaseModel):
    """Bounds on what one request can ask of the API."""

    max_message_length: int = Field(gt=0)
    token_lifetime_minutes: int = Field(gt=0)
    trace_lookup_timeout_s: float = Field(gt=0)
    graph_recursion_limit: int = Field(gt=0)
    postgres_timeout_s: float = Field(gt=0)


class SandboxLimits(BaseModel):
    """Bounds on generated Python, for RLM plans and `python_analysis` (rule 7)."""

    cpu_seconds: float = Field(gt=0)
    max_range_items: int = Field(gt=0)


class ResearchLimits(BaseModel):
    """Bounds on one RLM research run, so that a plan cannot run away (rule 7)."""

    max_depth: int = Field(gt=0)
    max_batches_per_level: int = Field(gt=0)
    max_sub_agent_calls: int = Field(gt=0)
    sub_agent_chunk_limit: int = Field(gt=0)
    max_parallel_sub_agents: int = Field(gt=0)
    max_search_calls: int = Field(gt=0)
    search_result_count: int = Field(gt=0)
    deadline_s: float = Field(gt=0)
    plan_regeneration_limit: int = Field(ge=0)


class ValidationLimits(BaseModel):
    """How many times the validator may send a turn back."""

    max_response_retries: int = Field(ge=0)
    max_evidence_retries: int = Field(ge=0)


class RoleRateLimit(BaseModel):
    """One role's token bucket."""

    capacity: int = Field(gt=0)
    refill_per_minute: float = Field(gt=0)


class ToolLimits(BaseModel):
    """Bounds on tool calls, so that a slow tool cannot hold a turn open."""

    search_timeout_s: float = Field(gt=0)
    analysis_timeout_s: float = Field(gt=0)
    mcp_timeout_s: float = Field(gt=0)
    mcp_token_lifetime_s: float = Field(gt=0)
    audit_timeout_s: float = Field(gt=0)
    max_rounds: int = Field(gt=0)
    max_calls_per_turn: int = Field(gt=0)
    max_result_characters: int = Field(gt=0)


class Limits(BaseModel):
    """All operational limits from `config/limits.yaml`."""

    chunking: ChunkingLimits
    indexing: IndexingLimits
    retrieval: RetrievalLimits
    circuit_breaker: CircuitBreakerLimits
    llm: LlmLimits
    conversation: ConversationLimits
    api: ApiLimits
    sandbox: SandboxLimits
    research: ResearchLimits
    tools: ToolLimits
    validation: ValidationLimits
    rate_limits: dict[str, RoleRateLimit]


class ModelNames(BaseModel):
    """Model identifiers from `config/models.yaml`."""

    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    rerank_model: str
    answer_effort: str
    # Maps a route name to its ordered model chain, such as "anthropic:claude-sonnet-5-5".
    llm_routes: dict[str, list[str]]


def _read_yaml_file(file_name: str) -> dict[str, object]:
    """Read one YAML file from the config directory."""
    with (CONFIG_DIRECTORY / file_name).open(encoding="utf-8") as yaml_file:
        parsed_content: dict[str, object] = yaml.safe_load(yaml_file)
    return parsed_content


class RoleAccess(BaseModel):
    """The document access levels of each role, and their order, from `config/rbac.yaml`."""

    roles: dict[str, tuple[str, ...]]
    role_ranking: tuple[str, ...]


@lru_cache
def get_environment_settings() -> EnvironmentSettings:
    """Return the environment settings, read once per process."""
    return EnvironmentSettings()


@lru_cache
def get_limits() -> Limits:
    """Return the operational limits, read once per process."""
    return Limits.model_validate(_read_yaml_file("limits.yaml"))


@lru_cache
def get_model_names() -> ModelNames:
    """Return the model identifiers, read once per process."""
    return ModelNames.model_validate(_read_yaml_file("models.yaml"))


@lru_cache
def get_role_access() -> RoleAccess:
    """Return the access levels of each role, read once per process."""
    return RoleAccess.model_validate(_read_yaml_file("rbac.yaml"))
