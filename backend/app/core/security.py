# name: security.py
# description: JWT token creation/verification and password hashing utilities.

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
import bcrypt

from app.core.config import settings

# ── Password Hashing ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Args:
        password: Plain-text password string.

    Returns:
        Bcrypt-hashed password string.
    """
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a bcrypt hash.

    Args:
        plain_password: User-submitted plain-text password.
        hashed_password: Stored bcrypt hash from the database.

    Returns:
        True if password matches, False otherwise.
    """
    pwd_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))


# ── JWT Tokens ────────────────────────────────────────────────────────────────
def create_access_token(data: dict[str, Any]) -> str:
    """
    Create a short-lived JWT access token.

    Args:
        data: Claims payload — must include 'sub' (user ID).

    Returns:
        Signed JWT access token string.
    """
    payload = data.copy()
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload.update({"exp": expire, "type": "access"})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(data: dict[str, Any]) -> str:
    """
    Create a long-lived JWT refresh token.

    Args:
        data: Claims payload — must include 'sub' (user ID).

    Returns:
        Signed JWT refresh token string.
    """
    payload = data.copy()
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload.update({"exp": expire, "type": "refresh"})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT string to decode.

    Returns:
        Decoded claims dict.

    Raises:
        JWTError: If the token is expired, malformed, or has an invalid signature.
    """
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:
        raise exc
