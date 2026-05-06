import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


logger = logging.getLogger(__name__)
configure_logging()


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()
    logger.info("Initializing FastAPI application")

    application = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured for %d origin entries", len(settings.cors_allow_origins))

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        logger.debug("Health check requested")
        return {"status": "ok"}

    application.include_router(api_router)
    logger.info("API routers registered")
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    logger.info("Starting uvicorn on %s:%s", settings.api_host, settings.api_port)
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port)
