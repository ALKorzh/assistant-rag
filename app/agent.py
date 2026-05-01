from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

from app.services.rag_service import RAGService


# 1. Определяем состояние
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]
    next_step: str
    is_relevant: str


# 2. Инициализируем модели
llm = ChatOllama(model="llama3.1", temperature=0)
llm_json = ChatOllama(model="llama3.1", format="json", temperature=0)
search_tool = DuckDuckGoSearchRun()


# 3. Узел: Роутер (ИСПРАВЛЕНО: добавлена логика вызова)
def router_node(state: AgentState):
    question = state["messages"][-1].content

    prompt = ChatPromptTemplate.from_template(
        """Ты — диспетчер интеллектуального помощника. Твоя задача — выбрать правильный путь.

        Запрос пользователя: {question}

        Реши, откуда взять информацию:
        - 'rag': если вопрос касается личных файлов, документов или сохраненных данных пользователя.
        - 'web_search': если нужен интернет (новости, текущие события, факты о мире).
        - 'direct': если это простое приветствие, благодарность или общая болтовня, не требующая поиска.

        Ответь ТОЛЬКО в формате JSON:
        {{"next_step": "rag"}} ИЛИ {{"next_step": "web_search"}} ИЛИ {{"next_step": "direct"}}
        """
    )

    # Цепочка: Промпт -> Модель -> Парсер JSON
    chain = prompt | llm_json | JsonOutputParser()

    try:
        result = chain.invoke({"question": question})
        return {"next_step": result["next_step"]}
    except Exception as e:
        print(f"Ошибка роутера: {e}")
        return {"next_step": "direct"}


# 4. Узел: RAG
def rag_node(state: AgentState):
    question = state["messages"][-1].content
    rag_service = RAGService()
    context = rag_service.query(question)
    # Возвращаем контекст как системное сообщение для генератора
    return {"messages": [HumanMessage(content=f"Используй этот контекст из документов для ответа: {context}")]}


# 5. Узел: Поиск
def search_node(state: AgentState):
    question = state["messages"][-1].content
    search_result = search_tool.run(question)
    return {"messages": [HumanMessage(content=f"Используй результаты поиска из интернета: {search_result}")]}


# 6. Узел: Генерация ответа
def generate_answer_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 7. Узел: Оценщик релевантности
def relevance_grader_node(state: AgentState):
    question = state["messages"][0].content  # Исходный вопрос
    # Последнее сообщение в списке — это контекст из RAG
    context = state["messages"][-1].content

    prompt = ChatPromptTemplate.from_template(
        """Ты — эксперт-оценщик. Твоя задача: определить, содержит ли найденный контекст ответ на вопрос.

        Вопрос: {question}
        Контекст: {context}

        Если в контексте есть хотя бы намек на ответ, ответь "yes".
        Если контекст абсолютно не связан с вопросом, ответь "no".

        Ответь ТОЛЬКО в формате JSON: {{"score": "yes" | "no"}}
        """
    )

    chain = prompt | llm_json | JsonOutputParser()
    result = chain.invoke({{"question": question, "context": context}})

    return {{"is_relevant": result["score"]}}


# --- Сборка графа ---
workflow = StateGraph(AgentState)

# Добавляем узлы
workflow.add_node("router", router_node)
workflow.add_node("rag", rag_node)
workflow.add_node("grader", relevance_grader_node) # Новый узел!
workflow.add_node("search", search_node)
workflow.add_node("generator", generate_answer_node)

workflow.set_entry_point("router")

# 1. Роутер решает: в RAG, в Поиск или Сразу отвечать
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_step"],
    {
        "rag": "rag",
        "web_search": "search",
        "direct": "generator"
    }
)

# 2. ПОСЛЕ RAG всегда идем в Grader
workflow.add_edge("rag", "grader")

# 3. Grader решает: достаточно инфы или идти в интернет
workflow.add_conditional_edges(
    "grader",
    lambda x: x["is_relevant"],
    {
        "yes": "generator",
        "no": "search"
    }
)

workflow.add_edge("search", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()