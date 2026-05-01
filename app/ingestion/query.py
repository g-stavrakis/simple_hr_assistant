from __future__ import annotations

import sys

from qdrant_client import QdrantClient

from app.config import get_settings
from app.rag.utils.tools import close_retriever, get_retriever


def extract_display_fields(metadata: dict, content: str) -> tuple[str, str, str, str]:
    # Pull the fields we want to display from either stored metadata or raw content.
    source_path = metadata.get("source_path", metadata.get("source", "unknown"))
    category = metadata.get("category", "unknown")
    document_name = metadata.get("document_name", "unknown")
    preview = content[:300].replace("\n", " ")
    return source_path, category, document_name, preview


def print_documents(documents) -> None:
    # Print retrieval results returned by the LangChain retriever.
    print("\nMATCHED CHUNKS:\n")

    for i, doc in enumerate(documents, start=1):
        source_path, category, document_name, preview = extract_display_fields(
            doc.metadata,
            doc.page_content,
        )

        print(f"{i}. category={category}")
        print(f"   document={document_name}")
        print(f"   source={source_path}")
        print(f"   preview={preview}...")
        print("=" * 40)


def print_all_indexed_chunks() -> None:
    settings = get_settings()
    client = None
    try:
        # Open the local Qdrant collection directly for full inspection.
        client = QdrantClient(path=settings.qdrant_path)
    except RuntimeError as exc:
        if "already accessed by another instance of Qdrant client" in str(exc):
            raise RuntimeError(
                f"Qdrant local storage at '{settings.qdrant_path}' is locked by another running process. "
                "Stop any active `rag-chat`, `rag-search`, notebook, debugger, or Python shell using the same local database and try again. "
                "If you need concurrent access, run Qdrant as a server instead of local mode."
            ) from exc
        raise

    print("\nQUERY:\n")
    print("ALL DOCUMENTS")
    print("\nINDEXED CHUNKS:\n")

    offset = None
    counter = 1

    while True:
        # Scroll through the whole collection in batches.
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=50,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            content = payload.get("page_content", "")
            source_path, category, document_name, preview = extract_display_fields(
                metadata,
                content,
            )

            print(f"{counter}. category={category}")
            print(f"   document={document_name}")
            print(f"   source={source_path}")
            print(f"   preview={preview}...")
            print("=" * 40)
            counter += 1

        if offset is None:
            break

    if client is not None:
        client.close()


def main() -> None:
    # Without a query, show the whole collection; otherwise run retrieval only.
    query = sys.argv[1] if len(sys.argv) > 1 else None

    if not query:
        print_all_indexed_chunks()
        return

    retriever = get_retriever()
    try:
        documents = retriever.invoke(query)
    finally:
        # Release the temporary retriever client after the lookup.
        close_retriever(retriever)

    print("\nQUERY:\n")
    print(query)
    print_documents(documents)


if __name__ == "__main__":
    main()
