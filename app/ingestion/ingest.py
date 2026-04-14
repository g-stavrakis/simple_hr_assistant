from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.vectorstore import resolve_retrieval_config, retrieval_mode_label


DATA_DIR = Path("data/docs")
SUPPORTED_TEXT_GLOBS = ["**/*.md", "**/*.txt"]
SUPPORTED_PDF_GLOB = "**/*.pdf"


def load_documents():
    docs = []

    for pattern in SUPPORTED_TEXT_GLOBS:
        loader = DirectoryLoader(
            str(DATA_DIR),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=False,
        )
        docs.extend(loader.load())

    pdf_loader = DirectoryLoader(
        str(DATA_DIR),
        glob=SUPPORTED_PDF_GLOB,
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=False,
    )
    docs.extend(pdf_loader.load())

    if not docs:
        raise ValueError(
            "No documents found in data/docs. Add at least one .md, .txt, or .pdf file."
        )

    return docs


def main() -> None:
    settings = get_settings()

    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    dense_embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    retrieval_mode, extra_kwargs = resolve_retrieval_config()
    qdrant_path = Path(settings.qdrant_path)

    # Recreate the local database for a deterministic demo run.
    if qdrant_path.exists():
        shutil.rmtree(qdrant_path)

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense_embeddings,
        retrieval_mode=retrieval_mode,
        path=str(qdrant_path),
        collection_name=settings.qdrant_collection,
        **extra_kwargs,
    )

    print(
        f"Indexed {len(chunks)} chunks into '{settings.qdrant_collection}' "
        f"using {retrieval_mode_label(retrieval_mode)} retrieval."
    )


if __name__ == "__main__":
    main()
