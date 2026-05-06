from pydantic import BaseModel, Field


class Query(BaseModel):
    """Inbound chat request body."""

    text: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    """Outbound chat answer body."""

    answer: str


class UploadResponse(BaseModel):
    """Result of indexing a document."""

    status: str
