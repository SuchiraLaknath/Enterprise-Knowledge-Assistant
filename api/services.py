"""The long-lived objects the API uses, created once at startup."""

from dataclasses import dataclass

from agent.dependencies import AgentDependencies
from agent.graph import AgentGraph
from api.checkpoint_store import CheckpointStore
from api.ratelimit import RateLimiter
from config.settings import ApiLimits, RoleAccess


@dataclass(frozen=True)
class ApplicationServices:
    """Held on `app.state`, so that tests can supply fakes without any patching."""

    agent_graph: AgentGraph
    # The same graph with in-memory checkpoints, used for a turn when Postgres is down.
    in_memory_graph: AgentGraph
    checkpoint_store: CheckpointStore
    agent_dependencies: AgentDependencies
    jwt_secret: str
    role_access: RoleAccess
    api_limits: ApiLimits
    langsmith_project: str
    rate_limiter: RateLimiter
