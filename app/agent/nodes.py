from functools import lru_cache

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.grader import advanced_keyword_check
from app.agent.llm import get_answer_llm, get_router_llm, invoke_router_text
from app.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    RAG_CONTEXT_PREFIX,
    ROUTER_SYSTEM_PROMPT,
    WEATHER_EXTRACT_PROMPT,
    WEB_SEARCH_REWRITE_PROMPT,
    WIKIPEDIA_QUERY_PROMPT,
    YOUTUBE_QUERY_PROMPT,
)
from app.agent.state import AgentState, RelevanceFlag, RouteResponse
from app.services.rag_service import RAGService
from app.tools.weather import get_weather
from app.tools.wiki_tool import get_wikipedia_info
from app.tools.youtube_tool import search_youtube_videos


@lru_cache(maxsize=1)
def _search_tool() -> DuckDuckGoSearchRun:
    return DuckDuckGoSearchRun()


@lru_cache(maxsize=1)
def _rag_service() -> RAGService:
    return RAGService()


def _message_text(message: BaseMessage) -> str:
    return str(message.content).strip()


def _last_message_text(state: AgentState) -> str:
    return _message_text(state["messages"][-1])


def _first_message_text(state: AgentState) -> str:
    return _message_text(state["messages"][0])


def router_node(state: AgentState) -> dict[str, str]:
    print("\n[STEP 1: ROUTER] Анализ интента...")
    structured_llm = get_router_llm().with_structured_output(RouteResponse)

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=_last_message_text(state)),
            ]
        )
        print(f"--- РЕШЕНИЕ: {result.next_step} | {result.reason} ---")
        return {"next_step": result.next_step, "is_relevant": "yes"}
    except Exception as exc:
        print(f"--- ОШИБКА РОУТЕРА: {exc} ---")
        return {"next_step": "direct", "is_relevant": "yes"}


def rag_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 2: RAG] Поиск в базе Qdrant...")
    context = _rag_service().query(_last_message_text(state))
    return {"messages": [AIMessage(content=f"{RAG_CONTEXT_PREFIX} {context}")]}


def relevance_grader_node(state: AgentState) -> dict[str, RelevanceFlag]:
    print("[STEP 3: ГРАДЕР] Оценка найденного контекста...")
    check = advanced_keyword_check(_first_message_text(state), _last_message_text(state))

    if not check["relevant"]:
        print(f"--- ГРАДЕР: ОТКЛОНЕНО ({check['reason']}) ---")
        return {"is_relevant": "no"}

    print(f"--- ГРАДЕР: ПРИНЯТО ({check['reason']}) ---")
    return {"is_relevant": "yes"}


def weather_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 2: WEATHER] Запрос метеоданных...")
    city = invoke_router_text(WEATHER_EXTRACT_PROMPT.format(message=_last_message_text(state)))
    result = get_weather(city)
    return {"messages": [AIMessage(content=f"ДАННЫЕ ПОГОДЫ: {result}")]}


def wikipedia_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 2: WIKIPEDIA] Поиск в энциклопедии...")
    query = invoke_router_text(WIKIPEDIA_QUERY_PROMPT.format(message=_last_message_text(state)))
    info = get_wikipedia_info(query)
    return {"messages": [AIMessage(content=f"СПРАВКА ВИКИПЕДИИ: {info}")]}


def youtube_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 2: YOUTUBE] Поиск видео...")
    query = invoke_router_text(YOUTUBE_QUERY_PROMPT.format(message=_last_message_text(state)))
    videos = search_youtube_videos(query)
    return {"messages": [AIMessage(content=f"РЕКОМЕНДАЦИИ YOUTUBE:\n{videos}")]}


def rewrite_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 3: REWRITE] Оптимизация для поиска...")
    query = invoke_router_text(WEB_SEARCH_REWRITE_PROMPT.format(message=_first_message_text(state)))
    return {"messages": [AIMessage(content=query)]}


def search_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 4: SEARCH] Поиск в сети...")
    result = _search_tool().run(_last_message_text(state))
    return {"messages": [AIMessage(content=f"Информация из сети: {result}")]}


def generate_answer_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    print("[STEP 5: GENERATOR] Синтез развернутого ответа...")
    is_relevant = state.get("is_relevant", "yes")
    final_messages: list[BaseMessage] = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]

    for message in state["messages"]:
        if (
            isinstance(message, AIMessage)
            and _message_text(message).startswith(RAG_CONTEXT_PREFIX)
            and is_relevant == "no"
        ):
            continue
        final_messages.append(message)

    response = get_answer_llm().invoke(final_messages)
    return {"messages": [response]}
