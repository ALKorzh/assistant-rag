from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    generate_answer_node,
    rag_node,
    relevance_grader_node,
    rewrite_node,
    router_node,
    search_node,
    weather_node,
    wikipedia_node,
    youtube_node,
)
from app.agent.state import AgentState


def build_graph():
    """Compile the LangGraph state machine that drives the assistant."""
    print("=" * 50)
    print("ЗАПУСК АГЕНТНОЙ СИСТЕМЫ...")

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

    compiled = workflow.compile()

    print("СИСТЕМА ГОТОВА")
    print("=" * 50)
    return compiled


app_graph = build_graph()
