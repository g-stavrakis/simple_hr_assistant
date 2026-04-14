from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.utils.state import RAGState
from app.rag.utils.tools import get_retriever, llm

retriever = get_retriever()


def _active_query(state: RAGState) -> str:
    if state.get("rewritten_question"):
        return state["rewritten_question"]
    else:
        return state["question"]


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4
    }


def _llm_grade_retrieval(question: str, docs: list[Document]) -> bool:
    context = "\n\n".join(doc.page_content[:500] for doc in docs[:3])
    prompt = ChatPromptTemplate.from_template(
        """
You are grading whether retrieved context is sufficient to answer a question.
Answer only yes or no.

Question: {question}

Retrieved context:
{context}

Is the context relevant and sufficient to answer the question?
""".strip()
    )
    decision = (prompt | llm).invoke({"question": question, "context": context}).content
    return decision.strip().lower().startswith("y")


def retrieve(state: RAGState) -> dict[str, Any]:
    query = _active_query(state)
    docs = retriever.invoke(query)
    return {"documents": docs}


def grade_retrieval(state: RAGState) -> dict[str, Any]:
    docs = state.get("documents", [])
    if not docs:
        return {"retrieval_ok": False}

    question_keywords = _keywords(state["question"])
    if not question_keywords:
        return {"retrieval_ok": _llm_grade_retrieval(state["question"], docs)}

    top_docs_text = " ".join(doc.page_content for doc in docs[:3]).lower()
    matched_keywords = sum(1 for keyword in question_keywords if keyword in top_docs_text)
    match_ratio = matched_keywords / len(question_keywords)
    enough_text = sum(len(doc.page_content) for doc in docs[:3]) >= 250

    if matched_keywords >= 2 or match_ratio >= 0.5:
        return {"retrieval_ok": enough_text}

    if matched_keywords == 0:
        return {"retrieval_ok": False}

    return {"retrieval_ok": _llm_grade_retrieval(state["question"], docs)}


def rewrite_query(state: RAGState) -> dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """
You rewrite user questions to improve retrieval for a hybrid RAG system.
Preserve the original meaning.
Make the query specific and keyword-rich.
Return only the rewritten query.

Original question: {question}
""".strip()
    )
    rewritten = (prompt | llm).invoke({"question": state["question"]}).content.strip()
    return {"rewritten_question": rewritten}


def generate(state: RAGState) -> dict[str, Any]:
    context = "\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(state["documents"])
    )
    prompt = ChatPromptTemplate.from_template(
        """
You are a concise assistant answering questions from retrieved context.
Use only the context below.
If the answer is not supported by the context, say you do not know.
When possible, mention the source number(s) you used.

Context:
{context}

Question:
{question}
""".strip()
    )
    answer = (prompt | llm).invoke(
        {"context": context, "question": state["question"]}
    ).content.strip()
    return {"answer": answer}


def route_after_grade(state: RAGState) -> str:
    if state["retrieval_ok"]:
        return "generate"
    if state.get("rewritten_question"):
        return "generate"
    return "rewrite_query"
