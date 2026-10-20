"""Authentication endpoints for the portal's cookie-backed sessions."""

from __future__ import annotations

import ipaddress
from time import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.config import settings
from app.auth import (
    SESSION_COOKIE,
    AuthenticatedUser,
    authenticate,
    authenticate_demo,
    create_session,
    current_csrf_token,
    invalidate_session,
    require_csrf,
    require_user,
    secure_cookie_enabled,
)
router = APIRouter(prefix="/api/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)
    role: Literal["employee", "admin"] | None = None


class AuthOptionsResponse(BaseModel):
    demo_login_enabled: bool


class UserResponse(BaseModel):
    username: str
    role: Literal["employee", "admin"]


class SessionResponse(BaseModel):
    user: UserResponse
    csrf_token: str


class LoginResponse(SessionResponse):
    pass


def _user_response(user: AuthenticatedUser) -> UserResponse:
    return UserResponse(username=user.username, role=user.role)  # type: ignore[arg-type]


def _demo_login_enabled_for(request: Request) -> bool:
    if not settings.demo_auth_enabled:
        return False
    if settings.demo_auth_public:
        return True

    client_host = request.client.host if request.client else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = False
    return settings.environment == "dev" and is_loopback


@router.get("/options", response_model=AuthOptionsResponse)
async def auth_options(request: Request) -> AuthOptionsResponse:
    return AuthOptionsResponse(demo_login_enabled=_demo_login_enabled_for(request))


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
) -> LoginResponse:
    user = authenticate(payload.username, payload.password)
    if user is not None and payload.role is not None and user.role != payload.role:
        user = None
    if user is None and payload.role and _demo_login_enabled_for(request):
        if not settings.demo_auth_public or payload.role == "employee":
            user = authenticate_demo(payload.role)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    session_token, csrf_token, expires_at = create_session(user)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_token,
        max_age=max(60, expires_at - int(time())),
        httponly=True,
        secure=secure_cookie_enabled(),
        samesite="lax",
        path="/",
    )
    return LoginResponse(user=_user_response(user), csrf_token=csrf_token)


@router.get("/session", response_model=SessionResponse)
async def session(request: Request, user: AuthenticatedUser = Depends(require_user)):
    csrf_token = current_csrf_token(
        request.cookies.get(SESSION_COOKIE),
        user.username,
    )
    if csrf_token is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return SessionResponse(user=_user_response(user), csrf_token=csrf_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    user: AuthenticatedUser = Depends(require_csrf),
) -> Response:
    del user
    invalidate_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        secure=secure_cookie_enabled(),
        httponly=True,
        samesite="lax",
    )
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
