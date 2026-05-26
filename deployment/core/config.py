from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    INTENT_LLM_MODEL: str = "groq/llama-3.1-8b-instant"
    GENERATION_LLM_MODEL: str = "groq/llama-3.3-70b-versatile"
    
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str
    
    RETRIEVAL_TOP_K: int = 3
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    
    class Config:
        extra = 'ignore'

config = Settings()
