from __future__ import annotations

import sys

from app.rag.agent import build_graph


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the employee probation period?"
    )

    graph = build_graph()
    result = graph.invoke(
        {
            "question": question,
            "rewritten_question": None,
            "documents": [],
            "retrieval_ok": False,
            "answer": None,
        }
    )

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
