"""Authentication with hardcoded demo users, which is option A in the assessment brief.

Rule 1 starts here: the role comes from this file and `config/rbac.yaml`, and it reaches the
graph as a verified `UserContext`. A token carries only the user's name, so a role can never
be claimed by a client. These demo passwords are not secrets, and all the data is mock data.
"""

import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.services import ApplicationServices
from retrieval.models import UserContext

TOKEN_ALGORITHM = "HS256"

bearer_token_reader = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class DemoUser:
    """One hardcoded user."""

    role: str
    password: str


DEMO_USERS = {
    "vimal": DemoUser(role="viewer", password="viewer-demo"),
    "nadee": DemoUser(role="analyst", password="analyst-demo"),
    "admin": DemoUser(role="admin", password="admin-demo"),
}


def get_services(request: Request) -> ApplicationServices:
    """Return the application's services."""
    services: ApplicationServices = request.app.state.services
    return services


def is_password_correct(username: str, password: str) -> bool:
    """Check a password in constant time, so that timing reveals nothing about it."""
    demo_user = DEMO_USERS.get(username)
    if demo_user is None:
        return False
    return hmac.compare_digest(password.encode(), demo_user.password.encode())


def create_access_token(username: str, services: ApplicationServices) -> str:
    """Return a signed token naming the user. The role is looked up on each request."""
    expires_at = datetime.now(UTC) + timedelta(minutes=services.api_limits.token_lifetime_minutes)
    return jwt.encode(
        {"sub": username, "exp": expires_at}, services.jwt_secret, algorithm=TOKEN_ALGORITHM
    )


def build_user_context(username: str, services: ApplicationServices) -> UserContext:
    """Create the verified identity that the graph receives."""
    role = DEMO_USERS[username].role
    return UserContext(
        user_id=username,
        role=role,
        allowed_access_levels=services.role_access.roles[role],
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_token_reader)],
    services: Annotated[ApplicationServices, Depends(get_services)],
) -> UserContext:
    """Verify the bearer token, and return the caller's identity."""
    invalid_login = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail="Please sign in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise invalid_login

    # Naming the algorithm stops a forged token from choosing a weaker one.
    try:
        token_claims = jwt.decode(
            credentials.credentials,
            services.jwt_secret,
            algorithms=[TOKEN_ALGORITHM],
            # A token with no expiry would be valid forever, even if it was signed correctly.
            options={"require": ["exp", "sub"]},
        )
    except jwt.InvalidTokenError as error:
        raise invalid_login from error

    username = token_claims.get("sub")
    if username not in DEMO_USERS:
        raise invalid_login
    return build_user_context(username, services)
