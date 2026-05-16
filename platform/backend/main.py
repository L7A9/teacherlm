from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from db.session import dispose_engine
from dispatcher.registry import get_registry
from routers import chat, conversations, files, generate, generators, health
from services.retrieval_orchestrator import get_retrieval_orchestrator
from services.storage_service import get_storage
from services.vector_service import get_vector_service


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("starting %s (env=%s)", settings.app_name, settings.environment)

    # Warm-load the generator registry — fail fast if the JSON is missing or malformed.
    get_registry().reload()

    await get_storage().ensure_bucket()
    await get_retrieval_orchestrator().warmup()

    app.state.arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    logger.info("connected to redis at %s", settings.redis_url)

    try:
        yield
    finally:
        logger.info("shutting down %s", settings.app_name)
        pool = getattr(app.state, "arq_pool", None)
        if pool is not None:
            await pool.close()
        await get_vector_service().aclose()
        await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(files.router)
    app.include_router(generators.router)
    app.include_router(chat.router)
    app.include_router(generate.router)

    return app


app = create_app()
