import os
import re
from typing import Annotated, Literal, TypedDict

import pymorphy3
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService
from app.tools.weather import get_weather
from app.tools.wiki_tool import get_wikipedia_info
from app.tools.youtube_tool import search_youtube_videos


GEMINI_MODEL = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RAG_CONTEXT_PREFIX = "Контекст из документов:"

RouteName = Literal["rag", "weather", "web_search", "wikipedia", "youtube", "direct"]
RelevanceFlag = Literal["yes", "no"]

ROUTER_SYSTEM_PROMPT = """Ты — интеллектуальный диспетчер.
- 'rag': если вопрос касается личных документов, лабораторных работ, конспектов.
- 'weather': если вопрос про погоду в городах.
- 'wikipedia': если нужны энциклопедические факты, история, биографии.
- 'youtube': если пользователь хочет найти видео или уроки.
- 'web_search': если нужны новости или общие факты из сети.
- 'direct': если это приветствие, вопрос по истории чата или просто разговор.
"""

ANSWER_SYSTEM_PROMPT = (
    "Ты — полезный и вежливый личный ассистент. Твои ответы должны быть подробными, "
    "структурированными и учитывать всю историю диалога. Если тебе предоставлены данные "
    "(из документов, википедии или поиска), обязательно используй их в ответе."
)


print("=" * 50)
print("ЗАПУСК АГЕНТНОЙ СИСТЕМЫ...")

gemini_router = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)

search_tool = DuckDuckGoSearchRun()
morph = pymorphy3.MorphAnalyzer()

print("СИСТЕМА ГОТОВА")
print("=" * 50)


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


def _message_text(message: BaseMessage) -> str:
    """Return message content as plain text for prompt/tool calls."""
    return str(message.content).strip()


def _last_message_text(state: AgentState) -> str:
    return _message_text(state["messages"][-1])


def _first_message_text(state: AgentState) -> str:
    return _message_text(state["messages"][0])


def _invoke_gemini_text(prompt: str) -> str:
    response = gemini_router.invoke(prompt)
    return str(response.content).strip()


def _get_lemmas(text: str) -> list[str]:
    tokens = re.findall(r"\b[а-яёa-z0-9]{2,}\b", text.lower())
    return [morph.parse(token)[0].normal_form for token in tokens]


def advanced_keyword_check(question: str, context: str) -> RelevanceCheck:
    """Fast lexical check that rejects obviously irrelevant RAG context."""
    question_lemmas = _get_lemmas(question)
    context_lemmas = _get_lemmas(context)
    question_numbers = set(re.findall(r"\b\d+\b", question))
    context_numbers = set(re.findall(r"\b\d+\b", context))

    if question_numbers and not question_numbers.intersection(context_numbers):
        return {"relevant": False, "reason": "Числовое несовпадение"}

    education_terms = {"лабораторный", "работа", "задание", "курс", "тема"}
    if any(term in question_lemmas for term in education_terms) and not any(
        term in context_lemmas for term in education_terms
    ):
        return {"relevant": False, "reason": "Нет учебной лексики"}

    keywords = [lemma for lemma in question_lemmas if len(lemma) > 3 and not lemma.isdigit()]
    matches = [lemma for lemma in keywords if lemma in context_lemmas]
    score = len(matches) / len(keywords) if keywords else 1.0
    return {"relevant": score >= 0.3, "reason": f"Score: {score:.2f}"}


def router_node(state: AgentState) -> dict[str, str]:
    print("\n[STEP 1: ROUTER] Анализ интента...")
    structured_llm = gemini_router.with_structured_output(RouteResponse)

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
    context = RAGService().query(_last_message_text(state))
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
    city = _invoke_gemini_text(
        "Extract city name in English from this user request. "
        f"Output only the city name: {_last_message_text(state)}"
    )
    result = get_weather(city)
    return {"messages": [AIMessage(content=f"ДАННЫЕ ПОГОДЫ: {result}")]}


def wikipedia_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 2: WIKIPEDIA] Поиск в энциклопедии...")
    query = _invoke_gemini_text(
        "Create a concise Russian Wikipedia search query for this request. "
        f"Output only the query: {_last_message_text(state)}"
    )
    info = get_wikipedia_info(query)
    return {"messages": [AIMessage(content=f"СПРАВКА ВИКИПЕДИИ: {info}")]}


def youtube_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 2: YOUTUBE] Поиск видео...")
    query = _invoke_gemini_text(
        "Create a concise YouTube search query for this request. "
        f"Output only the query: {_last_message_text(state)}"
    )
    videos = search_youtube_videos(query)
    return {"messages": [AIMessage(content=f"РЕКОМЕНДАЦИИ YOUTUBE:\n{videos}")]}


def rewrite_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 3: REWRITE] Оптимизация для поиска...")
    query = _invoke_gemini_text(
        "Rewrite this user question as a concise web search query. "
        f"Output only the query: {_first_message_text(state)}"
    )
    return {"messages": [AIMessage(content=query)]}


def search_node(state: AgentState) -> dict[str, list[AIMessage]]:
    print("[STEP 4: SEARCH] Поиск в сети...")
    result = search_tool.run(_last_message_text(state))
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

    response = llm.invoke(final_messages)
    return {"messages": [response]}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("grader", relevance_grader_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("wikipedia", wikipedia_node)
    workflow.add_node("youtube", youtube_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("search", search_node)
    workflow.add_node("generator", generate_answer_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_step"],
        {
            "rag": "rag",
            "weather": "weather",
            "wikipedia": "wikipedia",
            "youtube": "youtube",
            "web_search": "rewrite",
            "direct": "generator",
        },
    )
    workflow.add_conditional_edges(
        "grader",
        lambda state: state["is_relevant"],
        {
            "yes": "generator",
            "no": "rewrite",
        },
    )

    workflow.add_edge("weather", "generator")
    workflow.add_edge("wikipedia", "generator")
    workflow.add_edge("youtube", "generator")
    workflow.add_edge("rag", "grader")
    workflow.add_edge("rewrite", "search")
    workflow.add_edge("search", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()


app_graph = build_graph()
