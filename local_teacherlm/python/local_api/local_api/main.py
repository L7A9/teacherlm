from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from local_api.config import get_settings
from local_api.db import get_store
from local_api.services.coursebuilder import get_coursebuilder_service
from local_api.services.ingestion import get_ingestion_service
from local_api.routers import (
    artifacts,
    chat,
    conversations,
    coursebuilder,
    files,
    generate,
    generators,
    health,
    indexes,
    knowledge_checks,
    knowledge_graph,
    mcp,
    review_tests,
    setup,
    settings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app_settings = get_settings()
    logger.info("starting %s at %s", app_settings.app_name, app_settings.data_dir)
    get_store().initialize()
    get_ingestion_service().resume_incomplete_uploads()
    get_coursebuilder_service().resume_incomplete_builds()
    yield
    logger.info("shutting down %s", app_settings.app_name)


def create_app() -> FastAPI:
    app_settings = get_settings()
    app = FastAPI(title=app_settings.app_name, debug=app_settings.debug, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
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
    app.include_router(settings.router)
    app.include_router(artifacts.router)
    app.include_router(coursebuilder.router)
    app.include_router(knowledge_checks.router)
    app.include_router(knowledge_graph.router)
    app.include_router(indexes.router)
    app.include_router(review_tests.router)
    app.include_router(mcp.router)
    app.include_router(setup.router)
    return app


app = create_app()
