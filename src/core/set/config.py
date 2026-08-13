from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str | None = None
    LLM_MODEL: str | None = None
    LLM_TIMEOUT: float = 30.0
    LLM_TEMPERATURE: float = 0.1
    HF_TOKEN: str | None = None
    RAG_STORAGE_DIR: str = "storage/rag"
    RAG_CHROMA_DIR: str = "storage/chroma_db"
    RAG_COLLECTION_NAME: str = "default"
    RAG_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    RAG_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 80
    RAG_MIN_CHUNK_SIZE: int = 80
    RAG_RETRIEVE_TOP_K: int = 5
    RAG_RERANK_TOP_K: int = 3
    RAG_MAX_UPLOAD_MB: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


def require_llm_settings() -> Settings:
    missing = [
        name
        for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "LLM_MODEL")
        if not getattr(settings, name)
    ]
    if missing:
        raise RuntimeError(
            "missing required LLM settings: " + ", ".join(missing)
        )
    return settings
