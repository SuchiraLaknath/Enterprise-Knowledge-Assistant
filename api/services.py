"""The long-lived objects the API uses, created once at startup."""

from dataclasses import dataclass

from agent.dependencies import AgentDependencies
from agent.graph import AgentGraph
from api.ratelimit import RateLimiter
from config.settings import ApiLimits, RoleAccess


@dataclass(frozen=True)
class ApplicationServices:
    """Held on `app.state`, so that tests can supply fakes without any patching."""

    agent_graph: AgentGraph
    agent_dependencies: AgentDependencies
    jwt_secret: str
    role_access: RoleAccess
    api_limits: ApiLimits
    langsmith_project: str
    rate_limiter: RateLimiter
