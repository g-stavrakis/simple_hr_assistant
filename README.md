# Minimal Hybrid RAG with LangGraph, OpenAI, Qdrant, and LangSmith

This project is a small, runnable RAG example that uses:

- **LangGraph** for orchestration
- **OpenAI** for chat and dense embeddings
- **Qdrant** in **local mode** for retrieval
- **Optional FastEmbed BM25** for sparse retrieval on supported platforms
- **Dense retrieval by default**, with optional hybrid search via Qdrant dense + sparse retrieval
- **LangSmith** via environment variables for tracing

## Project structure

```text
rag_hybrid_qdrant/
├── app/
│   ├── config.py
│   ├── ingestion/
│   │   ├── ingest.py
│   │   └── vectorstore.py
│   └── rag/
│       ├── utils/
│       │   ├── tools.py
│       │   ├── nodes.py
│       │   └── state.py
│       ├── agent.py
│       └── main.py
├── data/docs/
│   └── example_policy.md
├── .env.example
├── pyproject.toml
└── README.md
```

## 1. Install uv

```bash
brew install uv
```

Other install options are available in the official uv docs if you are not on macOS.

## 2. Use Python 3.11

```bash
uv python install 3.11
uv sync
```

The project is pinned to Python 3.11 via `.python-version`. This avoids slow or failing dependency resolution on older local interpreters such as Python 3.9.

`uv sync` creates a `.venv` automatically.

If you want hybrid dense+sparse retrieval on a platform that supports `fastembed`, install the optional extra:

```bash
uv sync --extra hybrid
```

On Intel macOS, `uv` may fail to resolve `onnxruntime` for `fastembed`. In that case, stay on the default install and the app will use dense retrieval automatically.

## 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and add your real keys.

Minimum setup:

```env
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-hybrid-qdrant-demo
```

## 4. Ingest documents

```bash
uv run rag-ingest
```

This creates a local Qdrant database directory and indexes the files from `data/docs/`.
The command prints whether it indexed in `dense` or `hybrid` mode.

## 5. Run the app

```bash
uv run rag-chat "What is the employee probation period?"
```

Example output will include:
- the final answer
- the retrieved source previews

You can also run the modules directly:

```bash
uv run python -m app.ingestion.ingest
uv run python -m app.rag.main "What is the employee probation period?"
```

## How the graph works

The graph is intentionally small:

1. `retrieve`
2. `grade_retrieval`
3. either:
   - `generate`, or
   - `rewrite_query` and then retrieve once more

This is enough to show useful LangSmith traces without making the demo complicated.
