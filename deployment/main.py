from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load env variables at the top before core configuration is imported
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from deployment.api.routes import router  # noqa: E402
from deployment.services import (  # noqa: E402
    emotion_detection,
    language_detection,
    rag_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Emotion Detection Model...")
    emotion_detection.load_emotion_model()

    print("Loading Language Detection Model...")
    language_detection.load_language_model()

    print("Initializing RAG (SentenceTransformers & Qdrant)...")
    rag_service.init_rag()

    print("All backend services loaded successfully!")
    yield
    print("Shutting down and cleaning up resources...")


app = FastAPI(title="Mental Health RAG Chatbot API", lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Mental Health RAG Chatbot API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


app.include_router(router)
