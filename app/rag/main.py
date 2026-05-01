from __future__ import annotations

import sys

from app.rag.agent import build_graph
from app.rag.utils.tools import close_cached_vectorstore


def main() -> None:
    # Accept a CLI question, or fall back to a simple demo prompt.
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the employee probation period?"
    )

    graph = build_graph()
    try:
        # Seed the graph with the state fields used across the workflow.
        result = graph.invoke(
            {
                "question": question,
                "intent": None,
                "category": None,
                "rewritten_question": None,
                "documents": [],
                "answer": None,
                "messages": [],
            }
        )
    finally:
        # Release the cached Qdrant client after the request finishes.
        close_cached_vectorstore()

    # Print the final response and the retrieved source previews.
    print("\nQUESTION:\n")
    print(question)
    print("\nANSWER:\n")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:\n")
    for i, doc in enumerate(result["documents"], start=1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:220].replace("\n", " ")
        print(f"{i}. {source} :: {preview}...")


if __name__ == "__main__":
    main()
