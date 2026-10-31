"""The FastAPI application: startup, logging and routes.

Run with: uv run uvicorn api.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent.dependencies import AgentDependencies
from agent.graph import build_agent_graph, create_checkpoint_serializer
from agent.llm_gateway import LlmGateway
from api.ratelimit import build_rate_limiter
from api.routes import auth, chat
from api.services import ApplicationServices
from api.tracing import configure_redacted_tracing
from config.settings import (
    BM25_PARAMETERS_PATH,
    DOCUMENT_CATALOG_PATH,
    DOCUMENTS_DIRECTORY,
    EnvironmentSettings,
    export_tracing_settings,
    get_environment_settings,
    get_limits,
    get_model_names,
    get_role_access,
)
from resilience.circuit_breaker import CircuitBreaker
from retrieval.bm25_fallback import LocalKeywordIndex
from retrieval.catalog import DocumentCatalog
from retrieval.hybrid import load_bm25_encoder
from retrieval.pinecone_client import PineconeSearchBackend
from retrieval.search import DocumentSearchService
from tools.audit_log import PostgresAuditLog
from tools.mcp_client import McpClient
from tools.registry import build_tool_registry

logger = structlog.get_logger()

# Shorter secrets can be guessed offline from a captured token, which would forge any login.
MINIMUM_JWT_SECRET_BYTES = 32


def configure_logging() -> None:
    """Write structured JSON logs, including the ids bound to each request."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


def require_startup_secrets(settings: EnvironmentSettings) -> None:
    """Refuse to start without secrets, because a weak JWT secret would allow forged tokens."""
    if len(settings.jwt_secret.get_secret_value().encode()) < MINIMUM_JWT_SECRET_BYTES:
        raise RuntimeError("JWT_SECRET is too short. Generate one: openssl rand -hex 32")
    if not settings.anthropic_api_key.get_secret_value():
        raise RuntimeError("ANTHROPIC_API_KEY must be set.")
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set.")


def create_connection_pool(
    database_url: str,
) -> AsyncConnectionPool[AsyncConnection[dict[str, Any]]]:
    """Create the Postgres pool, with the connection settings the checkpointer requires.

    A restart of Postgres kills every idle connection, and the pool does not notice by itself.
    Testing each connection before lending it out means that the pool replaces a dead one,
    rather than failing the employee's turn.
    """
    return AsyncConnectionPool(
        database_url,
        connection_class=AsyncConnection[dict[str, Any]],
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the shared services at startup, and close their connections at shutdown."""
    configure_logging()
    settings = get_environment_settings()
    require_startup_secrets(settings)
    export_tracing_settings(settings)
    configure_redacted_tracing()
    limits = get_limits()
    model_names = get_model_names()

    # Step 1: load search data, which fails here rather than on the first question.
    sparse_encoder = load_bm25_encoder(BM25_PARAMETERS_PATH)
    local_keyword_index = LocalKeywordIndex.from_documents(
        DOCUMENTS_DIRECTORY, sparse_encoder, limits.chunking
    )

    # Step 2: open the connections for the lifetime of the application.
    connection_pool = create_connection_pool(settings.database_url)
    await connection_pool.open()
    checkpointer = AsyncPostgresSaver(connection_pool, serde=create_checkpoint_serializer())
    await checkpointer.setup()
    audit_log = PostgresAuditLog(connection_pool)
    await audit_log.setup()

    async with PineconeSearchBackend(settings, model_names) as search_backend:
        # Step 3: assemble the services that the graph uses.
        document_search = DocumentSearchService(
            search_backend,
            sparse_encoder,
            local_keyword_index,
            CircuitBreaker(
                limits.circuit_breaker.failure_threshold,
                limits.circuit_breaker.recovery_seconds,
            ),
            limits.retrieval,
        )
        llm_gateway = LlmGateway(
            model_names.llm_routes,
            settings,
            limits.llm,
            limits.circuit_breaker,
            model_names.answer_effort,
        )
        app.state.services = ApplicationServices(
            agent_graph=build_agent_graph(checkpointer),
            agent_dependencies=AgentDependencies(
                llm=llm_gateway,
                document_search=document_search,
                document_catalog=DocumentCatalog.from_file(DOCUMENT_CATALOG_PATH),
                history_message_count=limits.conversation.history_message_count,
                research_limits=limits.research,
                sandbox_limits=limits.sandbox,
                tool_limits=limits.tools,
                tool_registry=build_tool_registry(
                    limits.tools, get_role_access().role_ranking, settings.enabled_faults
                ),
                audit_log=audit_log,
                mcp_client=McpClient(settings.mcp_url, settings.enabled_faults),
            ),
            jwt_secret=settings.jwt_secret.get_secret_value(),
            role_access=get_role_access(),
            api_limits=limits.api,
            langsmith_project=settings.langsmith_project,
            rate_limiter=build_rate_limiter(
                settings.redis_url, limits.rate_limits, settings.enabled_faults
            ),
        )
        logger.info("api_started")
        yield

    await connection_pool.close()


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    """Log an unexpected failure, and return a message that reveals no internal details."""
    logger.error("unhandled_error", path=request.url.path, error_type=type(error).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Something went wrong. Please try again."},
    )


def create_app(use_lifespan: bool = True) -> FastAPI:
    """Create the application. Tests skip the lifespan, and supply fake services instead."""
    application = FastAPI(
        title="Enterprise Knowledge Assistant", lifespan=lifespan if use_lifespan else None
    )
    application.add_exception_handler(Exception, handle_unexpected_error)
    application.include_router(auth.router)
    application.include_router(chat.router)

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Report that the API process is running."""
        return {"status": "ok"}

    return application


app = create_app()
