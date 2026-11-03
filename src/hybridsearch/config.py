from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """All runtime settings, loaded from .env or environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")   

    database_url: str = "postgresql://rag:rag@localhost:5433/hybridsearch"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    fts_config: str = "english"

settings = Settings()