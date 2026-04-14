from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    rewritten_question: str | None
    documents: list[Document]
    retrieval_ok: bool
    answer: str | None
