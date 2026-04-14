from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(..., alias="OPENAI_API_KEY", min_length=1)
    openai_chat_model: str = Field("gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    qdrant_path: str = Field("./qdrant_db", alias="QDRANT_PATH")
    qdrant_collection: str = Field("demo_docs", alias="QDRANT_COLLECTION")
    top_k: int = Field(4, alias="TOP_K")


@lru_cache
def get_settings() -> Settings:
    return Settings()
