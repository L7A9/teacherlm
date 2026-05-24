from __future__ import annotations

import logging
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from db.session import dispose_engine
from dispatcher.registry import get_registry
from routers import chat, conversations, course_player, coursebuilder, files, generate, generators, health, knowledge_checks, knowledge_graph, review_tests
from services.storage_service import get_storage


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("starting %s (env=%s)", settings.app_name, settings.environment)

    # Warm-load the generator registry — fail fast if the JSON is missing or malformed.
    get_registry().reload()

    try:
        await get_storage().ensure_bucket()
    except Exception:  # noqa: BLE001
        logger.exception("storage warmup failed; continuing so read-only routes can serve")

    app.state.retrieval_warmup_task = asyncio.create_task(_warm_retrieval())

    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        app.state.arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        logger.info("connected to redis at %s", settings.redis_url)
    except Exception:  # noqa: BLE001
        logger.exception("redis queue initialization failed; uploads will be unavailable")
        app.state.arq_pool = None

    try:
        yield
    finally:
        logger.info("shutting down %s", settings.app_name)
        pool = getattr(app.state, "arq_pool", None)
        if pool is not None:
            await pool.close()
        try:
            from services.vector_service import get_vector_service

            await get_vector_service().aclose()
        except Exception:  # noqa: BLE001
            logger.exception("vector service shutdown failed")
        await dispose_engine()


async def _warm_retrieval() -> None:
    try:
        from services.retrieval_orchestrator import get_retrieval_orchestrator

        await get_retrieval_orchestrator().warmup()
    except Exception:  # noqa: BLE001
        logger.exception("retrieval warmup failed; chat/generation will lazy-load retrieval")


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
        allow_origin_regex=settings.cors_origin_regex,
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
    app.include_router(knowledge_checks.router)
    app.include_router(knowledge_graph.router)
    app.include_router(review_tests.router)
    app.include_router(coursebuilder.router)
    app.include_router(course_player.router)

    return app


app = create_app()
