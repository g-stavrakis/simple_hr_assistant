from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.rag.utils.nodes import (
    generate,
    grade_retrieval,
    retrieve,
    rewrite_query,
    route_after_grade,
)
from app.rag.utils.state import RAGState


def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_retrieval", grade_retrieval)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "grade_retrieval")
    builder.add_conditional_edges(
        "grade_retrieval",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
