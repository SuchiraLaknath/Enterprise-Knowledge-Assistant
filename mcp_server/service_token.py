"""The signed token that carries the verified role to the MCP server.

Rule 2: the server must not believe a plain header, because anything that can reach its port
could send one. The API signs a short-lived token with the secret both share, so a role is only
accepted if the API vouched for it. It lives here because the client and the server both need it,
and the server runs apart from the rest of the application.
"""

from datetime import UTC, datetime, timedelta

import jwt

SERVICE_TOKEN_HEADER = "x-service-token"
TOKEN_ALGORITHM = "HS256"
# Stops a login token for the API from being accepted here, as both use the same secret.
TOKEN_AUDIENCE = "mcp-server"


def create_service_token(role: str, secret: str, lifetime_s: float) -> str:
    """Sign a token saying that the API verified this role."""
    expires_at = datetime.now(UTC) + timedelta(seconds=lifetime_s)
    return jwt.encode(
        {"role": role, "aud": TOKEN_AUDIENCE, "exp": expires_at}, secret, algorithm=TOKEN_ALGORITHM
    )


def read_verified_role(token: str, secret: str) -> str:
    """Return the role in a token, or an empty string if the token cannot be trusted."""
    if not token or not secret:
        return ""
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[TOKEN_ALGORITHM],
            audience=TOKEN_AUDIENCE,
            options={"require": ["exp", "aud", "role"]},
        )
    except jwt.InvalidTokenError:
        return ""
    return str(claims["role"])
