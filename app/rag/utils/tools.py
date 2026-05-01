from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.config import get_settings
from app.ingestion.vectorstore import resolve_retrieval_config

settings = get_settings()
# Shared chat model used by both retrieval nodes and the complaint-form agent.
llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0,
    api_key=settings.openai_api_key.get_secret_value(),
)


@lru_cache(maxsize=1)
def get_vectorstore():
    # Reuse one local Qdrant-backed vector store per process.
    dense_embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    retrieval_mode, extra_kwargs = resolve_retrieval_config()

    return QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        retrieval_mode=retrieval_mode,
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        **extra_kwargs,
    )


def build_category_filter(category: str | None) -> Filter | None:
    # Apply a metadata filter only when a category was detected.
    if not category:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="metadata.category",
                match=MatchValue(value=category),
            )
        ]
    )


def get_retriever(category: str | None = None, k: int | None = None):
    # Wrap the vector store as a retriever with optional category scoping.
    search_kwargs = {"k": k or settings.top_k}
    category_filter = build_category_filter(category)
    if category_filter is not None:
        search_kwargs["filter"] = category_filter

    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def close_retriever(retriever) -> None:
    # Close ad-hoc retriever clients so local Qdrant locks are released quickly.
    vectorstore = getattr(retriever, "vectorstore", None)
    client = getattr(vectorstore, "client", None)
    if client and hasattr(client, "close"):
        client.close()
    get_vectorstore.cache_clear()


def close_cached_vectorstore() -> None:
    # Safely close the shared cached vector store when the request completes.
    if get_vectorstore.cache_info().currsize == 0:
        return

    vectorstore = get_vectorstore()
    client = getattr(vectorstore, "client", None)
    if client and hasattr(client, "close"):
        client.close()
    get_vectorstore.cache_clear()


@tool
def get_complaint_form() -> str:
    """
    Return a simple complaint form template for employees.
    """
    # Keep the form deliberately simple so it can be reused in demos or UI prototypes.
    return """
Employee Complaint Form

1. Employee name:
2. Department:
3. Date of submission:
4. Type of complaint:
5. People involved:
6. Description of the incident:
7. Date and location of the incident:
8. Any witnesses:
9. Supporting evidence attached:
10. Requested resolution:
11. Employee signature:
12. HR follow-up notes:
""".strip()
