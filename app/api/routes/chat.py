import logging

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage

from app.api.deps import get_agent_graph
from app.schemas.chat import ChatResponse, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(query: Query, graph=Depends(get_agent_graph)) -> ChatResponse:
    """Run the user message through the agent graph and return the final answer."""
    logger.info("Chat request received, text_length=%d", len(query.text))
    initial_state = {"messages": [HumanMessage(content=query.text)]}
    result = await graph.ainvoke(initial_state)
    final_message = result["messages"][-1].content
    answer = str(final_message)
    logger.info("Chat response generated, answer_length=%d", len(answer))
    return ChatResponse(answer=answer)
