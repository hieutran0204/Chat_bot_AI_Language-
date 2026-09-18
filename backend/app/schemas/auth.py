# name: auth.py (schemas)
# description: Pydantic request/response schemas for authentication endpoints.

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Request body for user registration."""

    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=128)
    level: str = Field(default="A1", pattern=r"^(A1|A2|B1|B2|C1|C2)$")


class LoginRequest(BaseModel):
    """Request body for user login."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response body containing JWT token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """Request body for refreshing the access token."""

    refresh_token: str


class UserResponse(BaseModel):
    """Public user profile data returned after auth."""

    id: uuid.UUID
    email: str
    username: str
    level: str
    created_at: datetime

    model_config = {"from_attributes": True}
