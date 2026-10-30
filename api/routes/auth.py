"""Sign-in route."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import (
    DEMO_USERS,
    create_access_token,
    get_services,
    is_password_correct,
)
from api.services import ApplicationServices

router = APIRouter()


class LoginRequest(BaseModel):
    """A sign-in attempt."""

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(BaseModel):
    """A successful sign-in. The role is for display only, and is never read back."""

    access_token: str
    username: str
    role: str


@router.post("/auth/login")
async def login(
    request: LoginRequest,
    services: Annotated[ApplicationServices, Depends(get_services)],
) -> LoginResponse:
    """Exchange a username and password for a signed token."""
    if not is_password_correct(request.username, request.password):
        # The same message for both cases, so that it does not reveal which usernames exist.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")

    return LoginResponse(
        access_token=create_access_token(request.username, services),
        username=request.username,
        role=DEMO_USERS[request.username].role,
    )
