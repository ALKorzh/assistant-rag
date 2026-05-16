from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


RouteNameLiteral = Literal[
    "rag",
    "weather",
    "web_search",
    "wikipedia",
    "youtube",
    "direct",
    "calculator",
]
RelevanceFlag = Literal["yes", "no"]
ReflectionBranch = Literal["retry", "done"]


class RouteResponse(BaseModel):
    """Structured router decision returned by Gemini."""

    next_step: RouteNameLiteral = Field(description="Следующий узел агентного графа")
    reason: str = Field(description="Краткое обоснование выбора маршрута")


class AgentState(TypedDict, total=False):
    """LangGraph state shared by all agent nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_step: RouteNameLiteral
    is_relevant: RelevanceFlag
    reflection_branch: ReflectionBranch
    reflection_retry_count: int


class RelevanceCheck(TypedDict):
    relevant: bool
    reason: str


class AnswerReflection(BaseModel):
    """Structured review of an assistant draft answer."""

    score: int = Field(ge=1, le=5, description="Оценка качества ответа 1–5")
    feedback: str = Field(description="Что улучшить (кратко, на русском)")
