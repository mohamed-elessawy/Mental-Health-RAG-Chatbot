from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    GROQ_API_KEY: str = ""

    INTENT_LLM_MODEL: str = "groq/llama-3.1-8b-instant"
    GENERATION_LLM_MODEL: str = "groq/openai/gpt-oss-120b"

    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    RETRIEVAL_TOP_K: int = 3
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: list[str] = ["*"]


config = Settings()
