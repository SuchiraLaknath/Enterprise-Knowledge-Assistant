"""Typed access to configuration, so that a bad value fails at startup with a clear message.

Secrets come from the environment, and tunable values come from `config/*.yaml`. Each
build step adds the settings it needs here.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIRECTORY = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIRECTORY.parent

DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "docs"
BM25_PARAMETERS_PATH = PROJECT_ROOT / "data" / "bm25_params.json"
DOCUMENT_CATALOG_PATH = PROJECT_ROOT / "data" / "doc_catalog.json"


class EnvironmentSettings(BaseSettings):
    """Values that differ per machine or are secret, loaded from the environment or `.env`."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    pinecone_api_key: SecretStr
    pinecone_index: str
    # Comma-separated names of clients to fail on purpose, for the degradation demo.
    faults: str = ""

    @property
    def enabled_faults(self) -> set[str]:
        """Return the fault names, so callers can test membership instead of parsing text."""
        return {fault.strip() for fault in self.faults.split(",") if fault.strip()}


class ChunkingLimits(BaseModel):
    """Size rules for splitting documents into chunks."""

    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_fraction: float = Field(ge=0, lt=1)
    tokens_per_word: float = Field(gt=0)


class IndexingLimits(BaseModel):
    """Batch sizes and location for writing to Pinecone."""

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
    final_result_count: int = Field(gt=0)
    pinecone_timeout_s: float = Field(gt=0)
    rerank_timeout_s: float = Field(gt=0)


class CircuitBreakerLimits(BaseModel):
    """When to stop calling a failing service, and when to try it again."""

    failure_threshold: int = Field(gt=0)
    recovery_seconds: float = Field(gt=0)


class Limits(BaseModel):
    """All operational limits from `config/limits.yaml`."""

    chunking: ChunkingLimits
    indexing: IndexingLimits
    retrieval: RetrievalLimits
    circuit_breaker: CircuitBreakerLimits


class ModelNames(BaseModel):
    """Model identifiers from `config/models.yaml`."""

    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    rerank_model: str


def _read_yaml_file(file_name: str) -> dict[str, object]:
    """Read one YAML file from the config directory."""
    with (CONFIG_DIRECTORY / file_name).open(encoding="utf-8") as yaml_file:
        parsed_content: dict[str, object] = yaml.safe_load(yaml_file)
    return parsed_content


@lru_cache
def get_environment_settings() -> EnvironmentSettings:
    """Return the environment settings, read once per process."""
    return EnvironmentSettings()  # type: ignore[call-arg]  # pydantic fills fields from the env.


@lru_cache
def get_limits() -> Limits:
    """Return the operational limits, read once per process."""
    return Limits.model_validate(_read_yaml_file("limits.yaml"))


@lru_cache
def get_model_names() -> ModelNames:
    """Return the model identifiers, read once per process."""
    return ModelNames.model_validate(_read_yaml_file("models.yaml"))
