import logging

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage

from app.agent.nodes import extract_user_facing_answer
from app.api.deps import get_agent_graph
from app.api.security import require_api_key
from app.schemas.chat import ChatResponse, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, query: Query, graph=Depends(get_agent_graph)) -> ChatResponse:
    """Run the user message through the agent graph and return the final answer."""
    rid = getattr(request.state, "request_id", "-")
    logger.info("request_id=%s Chat request received, text_length=%d", rid, len(query.text))
    initial_state = {"messages": [HumanMessage(content=query.text)]}
    result = await graph.ainvoke(initial_state)
    answer = extract_user_facing_answer(result["messages"])
    if not answer.strip():
        answer = (
            "Не удалось получить текст ответа. Попробуйте уточнить вопрос или повторить запрос позже."
        )
    logger.info("request_id=%s Chat response generated, answer_length=%d", rid, len(answer))
    return ChatResponse(answer=answer)
