from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


RouteName = Literal["rag", "weather", "web_search", "wikipedia", "youtube", "direct"]
RelevanceFlag = Literal["yes", "no"]


class RouteResponse(BaseModel):
    """Structured router decision returned by Gemini."""

    next_step: RouteName = Field(description="Следующий узел агентного графа")
    reason: str = Field(description="Краткое обоснование выбора маршрута")


class AgentState(TypedDict, total=False):
    """LangGraph state shared by all agent nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_step: RouteName
    is_relevant: RelevanceFlag


class RelevanceCheck(TypedDict):
    relevant: bool
    reason: str
