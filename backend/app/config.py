from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 160
    top_k: int = 6
    rrf_k: int = 60
    min_relevance: float = 0.20
    max_context_chars: int = 12000
    max_upload_mb: int = 25
    cors_origins: str = "http://localhost:5173"

settings = Settings()
