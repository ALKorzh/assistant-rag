from fastapi import APIRouter

from app.api.routes import chat, upload


api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(upload.router)

__all__ = ["api_router"]
