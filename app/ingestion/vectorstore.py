from __future__ import annotations

import warnings
from typing import Any

from langchain_qdrant import QdrantVectorStore, RetrievalMode

try:
    from langchain_qdrant import FastEmbedSparse
except ImportError:
    FastEmbedSparse = None


def resolve_retrieval_config() -> tuple[RetrievalMode, dict[str, Any]]:
    if FastEmbedSparse is None:
        warnings.warn(
            "FastEmbed is not installed; falling back to dense retrieval. "
            "Install the optional 'hybrid' extra to enable dense+sparse retrieval.",
            stacklevel=2,
        )
        return RetrievalMode.DENSE, {}

    return (
        RetrievalMode.HYBRID,
        {"sparse_embedding": FastEmbedSparse(model_name="Qdrant/bm25")},
    )


def retrieval_mode_label(mode: RetrievalMode) -> str:
    return "hybrid" if mode == RetrievalMode.HYBRID else "dense"
