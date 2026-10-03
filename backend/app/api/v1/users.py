# name: users.py (API router)
# description: User profile and learning progress endpoints.

from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.security import hash_password
from app.models.progress import UserProgress
from app.models.user import User
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/users", tags=["Users"])


class UpdateProfileRequest(BaseModel):
    """Request body for updating user profile."""

    username: str | None = Field(default=None, min_length=3, max_length=100)
    level: str | None = Field(default=None, pattern=r"^(A1|A2|B1|B2|C1|C2)$")
    password: str | None = Field(default=None, min_length=8, max_length=128)


class ProgressResponse(BaseModel):
    """Daily progress entry."""

    date: date
    messages_sent: int
    vocab_learned: int
    study_minutes: int

    model_config = {"from_attributes": True}


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """
    Get the current user's profile.

    Args:
        current_user: Authenticated user from JWT.

    Returns:
        UserResponse with id, email, username, level, created_at.
    """
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    body: UpdateProfileRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the current user's profile fields.

    All fields are optional — only provided fields are updated.

    Args:
        body: Partial update payload.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        Updated UserResponse.
    """
    if body.username is not None:
        current_user.username = body.username
    if body.level is not None:
        current_user.level = body.level
    if body.password is not None:
        current_user.hashed_password = hash_password(body.password)

    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.get("/me/progress", response_model=list[ProgressResponse])
async def get_progress(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get learning progress for the last N days.

    Returns one record per day where the user was active.
    Ordered chronologically for charting in Flutter.

    Args:
        days: Number of days to look back (default 30, max 365).
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        List of ProgressResponse ordered by date ascending.
    """
    from datetime import UTC, datetime, timedelta

    if days > 365:
        days = 365

    cutoff = datetime.now(UTC).date() - timedelta(days=days)
    result = await db.execute(
        select(UserProgress)
        .where(
            UserProgress.user_id == current_user.id,
            UserProgress.date >= cutoff,
        )
        .order_by(UserProgress.date.asc())
    )
    return list(result.scalars().all())
