# name: users.py (API router)
# description: User profile and learning progress endpoints.

from datetime import UTC, date, datetime, timedelta

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


class WeaknessItemResponse(BaseModel):
    id: str
    weakness_type: str
    category: str
    original: str
    suggestion: str
    explanation: str | None = None
    is_repeated: bool = False
    created_at: datetime


class StrengthItemResponse(BaseModel):
    id: str
    strength_type: str
    category: str
    evidence: str
    confidence_score: float = 0.8
    created_at: datetime


class RadarScores(BaseModel):
    pronunciation: float
    grammar: float
    vocabulary: float
    fluency: float
    reflex: float


class UserAnalyticsResponse(BaseModel):
    user_id: str
    username: str
    email: str
    level: str
    rank_title: str
    cefr_level: str
    streak_days: int
    total_conversations: int
    total_messages: int
    total_user_messages: int
    accuracy_rate: float
    flawless_sessions: int
    radar_scores: RadarScores
    weaknesses: list[WeaknessItemResponse]
    strengths: list[StrengthItemResponse]


_WEAKNESS_CATEGORY_MAP = {
    "word_order": "Trật tự từ",
    "article_usage": "Mạo từ",
    "wrong_word": "Dùng sai từ",
    "tense_past_simple": "Thì Quá khứ đơn",
    "tense_present_perfect": "Thì HT Hoàn thành",
    "subject_verb_agreement": "Hòa hợp Chủ - Vị",
    "preposition": "Giới từ",
    "conditional": "Câu điều kiện",
    "false_friend": "Từ dễ nhầm",
    "collocation": "Cụm từ (Collocation)",
    "final_consonant": "Âm đuôi (Phát âm)",
    "th_sound": "Âm /th/ (Phát âm)",
}


@router.get("/me/analytics", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Aggregate comprehensive learning analytics for the current user in a single query.
    Calculates true streak days, accuracy rate, radar dimensions, and provides real
    recorded weaknesses and strengths without client-side N+1 loops.
    """
    from app.models.conversation import Conversation, Message
    from app.models.learner_analytics import UserStrengthLog, UserWeaknessLog
    from sqlalchemy import func

    uid = current_user.id

    # 1. Real recorded weaknesses from user_weakness_log
    weaknesses_res = await db.execute(
        select(UserWeaknessLog)
        .where(UserWeaknessLog.user_id == uid)
        .order_by(UserWeaknessLog.created_at.desc())
    )
    raw_weaknesses = weaknesses_res.scalars().all()

    weakness_items = [
        WeaknessItemResponse(
            id=str(w.id),
            weakness_type=w.weakness_type,
            category=_WEAKNESS_CATEGORY_MAP.get(w.weakness_type.lower(), w.weakness_type),
            original=w.original,
            suggestion=w.suggestion,
            explanation=w.explanation,
            is_repeated=w.is_repeated,
            created_at=w.created_at,
        )
        for w in raw_weaknesses
    ]

    # 2. Real recorded strengths from user_strength_log
    strengths_res = await db.execute(
        select(UserStrengthLog)
        .where(UserStrengthLog.user_id == uid)
        .order_by(UserStrengthLog.created_at.desc())
    )
    raw_strengths = strengths_res.scalars().all()

    strength_items = [
        StrengthItemResponse(
            id=str(s.id),
            strength_type=s.strength_type,
            category=_WEAKNESS_CATEGORY_MAP.get(s.strength_type.lower(), s.strength_type),
            evidence=s.evidence,
            confidence_score=s.confidence_score,
            created_at=s.created_at,
        )
        for s in raw_strengths
    ]

    # 3. Real Conversations & Messages statistics
    conv_cnt = (
        await db.execute(
            select(func.count(Conversation.id)).where(Conversation.user_id == uid)
        )
    ).scalar() or 0

    total_msg_cnt = (
        await db.execute(
            select(func.count(Message.id))
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == uid)
        )
    ).scalar() or 0

    user_msg_cnt = (
        await db.execute(
            select(func.count(Message.id))
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == uid, Message.role == "user")
        )
    ).scalar() or 0

    # Conversations with errors
    conv_err_res = await db.execute(
        select(func.count(func.distinct(UserWeaknessLog.conversation_id)))
        .where(UserWeaknessLog.user_id == uid)
    )
    convs_with_errors = conv_err_res.scalar() or 0
    flawless_sessions = max(0, conv_cnt - convs_with_errors)

    # Accuracy rate
    accuracy_rate = 100.0
    if user_msg_cnt > 0:
        error_msg_cnt = min(user_msg_cnt, len(raw_weaknesses))
        accuracy_rate = round(((user_msg_cnt - error_msg_cnt) / user_msg_cnt) * 100.0, 1)

    # 4. True consecutive streak days from user_progress
    progress_res = await db.execute(
        select(UserProgress)
        .where(UserProgress.user_id == uid)
        .order_by(UserProgress.date.desc())
    )
    progress_entries = progress_res.scalars().all()
    today = datetime.now(UTC).date()
    streak_days = 0
    expected = today
    for p in progress_entries:
        if p.date == expected:
            streak_days += 1
            expected -= timedelta(days=1)
        elif p.date == expected - timedelta(days=1) and streak_days == 0:
            streak_days += 1
            expected = p.date - timedelta(days=1)
        else:
            break

    total_study_minutes = sum(p.study_minutes for p in progress_entries)

    # 5. Radar dimensions computed dynamically from real error taxonomy
    grammar_types = {
        "tense_past_simple",
        "tense_present_perfect",
        "subject_verb_agreement",
        "article_usage",
        "preposition",
        "word_order",
        "conditional",
    }
    vocab_types = {"wrong_word", "false_friend", "collocation"}
    pronun_types = {"final_consonant", "th_sound"}

    grammar_errs = sum(1 for w in raw_weaknesses if w.weakness_type in grammar_types)
    vocab_errs = sum(1 for w in raw_weaknesses if w.weakness_type in vocab_types)
    pronun_errs = sum(1 for w in raw_weaknesses if w.weakness_type in pronun_types)

    level_base = {
        "A1": 55.0,
        "A2": 65.0,
        "B1": 75.0,
        "B2": 82.0,
        "C1": 88.0,
        "C2": 94.0,
    }.get(current_user.level, 55.0)

    grammar_score = round(max(25.0, min(95.0, level_base + 15.0 - (grammar_errs * 3.0))), 1)
    vocab_score = round(max(25.0, min(95.0, level_base + 15.0 - (vocab_errs * 4.0))), 1)
    pronun_score = round(max(25.0, min(95.0, level_base + 15.0 - (pronun_errs * 5.0))), 1)
    fluency_score = round(
        max(30.0, min(95.0, level_base + min(20.0, conv_cnt * 1.5 + total_study_minutes * 0.1))), 1
    )
    reflex_score = round(max(30.0, min(95.0, level_base + min(20.0, user_msg_cnt * 0.5))), 1)

    level_names = {
        "A1": "Beginner",
        "A2": "Elementary",
        "B1": "Intermediate",
        "B2": "Upper Intermediate",
        "C1": "Advanced",
        "C2": "Mastery",
    }
    lvl = current_user.level or "A1"
    cefr_level = f"{lvl} ({level_names.get(lvl, 'Learner')})"

    rank_title = "Tân Thủ Học Tiếng"
    if conv_cnt >= 30 and accuracy_rate >= 80:
        rank_title = "Chiến Tướng Giao Tiếp"
    elif conv_cnt >= 15:
        rank_title = "Cao Thủ Luyện Nói"
    elif conv_cnt >= 5:
        rank_title = "Tinh Anh Khởi Động"

    return UserAnalyticsResponse(
        user_id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        level=lvl,
        rank_title=rank_title,
        cefr_level=cefr_level,
        streak_days=streak_days,
        total_conversations=conv_cnt,
        total_messages=total_msg_cnt,
        total_user_messages=user_msg_cnt,
        accuracy_rate=accuracy_rate,
        flawless_sessions=flawless_sessions,
        radar_scores=RadarScores(
            pronunciation=pronun_score,
            grammar=grammar_score,
            vocabulary=vocab_score,
            fluency=fluency_score,
            reflex=reflex_score,
        ),
        weaknesses=weakness_items,
        strengths=strength_items,
    )
