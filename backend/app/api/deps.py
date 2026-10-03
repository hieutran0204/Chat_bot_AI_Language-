# name: deps.py
# description: Shared FastAPI dependencies — JWT authentication guard and current user resolution.

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select

from app.core.database import AsyncSessionFactory
from app.core.security import decode_token
from app.models.user import User

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> User:
    """
    FastAPI dependency: extract and validate the Bearer JWT token.

    Uses an isolated short session so DB connections are not held open
    across long-running SSE streaming responses.

    Args:
        credentials: HTTP Bearer credentials from the Authorization header.

    Returns:
        Authenticated User ORM instance.

    Raises:
        HTTPException 401: If the token is missing, invalid, expired,
                           or if the user no longer exists / is inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            raise credentials_exception
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = uuid.UUID(user_id_str)
    except (JWTError, ValueError):
        raise credentials_exception

    async with AsyncSessionFactory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_exception

    return user
