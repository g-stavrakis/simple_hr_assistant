from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import get_settings
from app.ingestion.vectorstore import resolve_retrieval_config

settings = get_settings()
llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0,
    api_key=settings.openai_api_key.get_secret_value(),
)


def get_retriever():
    dense_embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    retrieval_mode, extra_kwargs = resolve_retrieval_config()

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        retrieval_mode=retrieval_mode,
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        **extra_kwargs,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
