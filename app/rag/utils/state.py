from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    question: str
    intent: str | None
    category: str | None
    rewritten_question: str | None
    documents: list[Document]
    answer: str | None
    messages: list[BaseMessage]
