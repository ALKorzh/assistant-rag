from fastapi import FastAPI

from app.api.routes import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
    )
    application.include_router(api_router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port)
