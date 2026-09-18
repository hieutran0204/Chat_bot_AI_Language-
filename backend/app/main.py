# name: main.py
# description: FastAPI application entry point — registers routers, CORS middleware,
#              and modern async lifespan managing PostgreSQL and Redis connections.

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, chat, documents, users
from app.core.config import settings
from app.core.database import engine
from app.core.redis.client import get_redis_client

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan Context Manager ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    Initializes upload directory, establishes connection to self-hosted Redis,
    and handles graceful teardown of DB and cache connection pools.
    """
    # 1. Startup initialization
    os.makedirs(settings.upload_dir, exist_ok=True)
    redis_client = get_redis_client()
    await redis_client.connect()

    logger.info("=" * 60)
    logger.info("🚀 %s started", settings.app_name)
    logger.info("   Environment : %s", settings.app_env)
    logger.info("   LLM         : %s (%s)", settings.llm_provider, settings.ollama_chat_model)
    logger.info("   Embedder    : %s (%s)", settings.embedding_provider, settings.huggingface_embed_model)
    logger.info("   Database    : %s", settings.postgres_db)
    logger.info("   Redis       : %s", "Connected" if redis_client.is_available else "Offline (fallback enabled)")
    logger.info("=" * 60)

    yield

    # 2. Shutdown teardown
    await redis_client.disconnect()
    await engine.dispose()
    logger.info("🛑 %s stopped. Database & Redis connections released.", settings.app_name)


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    description="Voice-first AI English Speaking Tutor with local Redis & RAG capabilities.",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(users.router, prefix=API_PREFIX)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint reporting DB, Redis, and service status.

    Returns:
        JSON with health status metadata.
    """
    redis_client = get_redis_client()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "redis": "connected" if redis_client.is_available else "offline",
    }
