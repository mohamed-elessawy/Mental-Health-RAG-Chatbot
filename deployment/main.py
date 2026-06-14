from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)   

from deployment.api.routes import router  # noqa: E402
from deployment.core.config import config  # noqa: E402
from deployment.core.logging import setup_logging  # noqa: E402
from deployment.services import (  # noqa: E402
    emotion_detection,
    language_detection,
    rag_service,
)

logger = setup_logging(config.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading emotion detection model...")
    emotion_detection.load_emotion_model()
    logger.info("Emotion model loaded")

    logger.info("Loading language detection model...")
    language_detection.load_language_model()
    logger.info("Language model loaded")

    logger.info("Initializing RAG (embedder + Qdrant)...")
    rag_service.init_rag()
    logger.info("RAG initialized")

    logger.info("All services ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Serenity Mental Health Chatbot API", lifespan=lifespan)

# CORS: allow the GitHub Pages frontend and localhost for local development.
DEFAULT_ORIGINS = [
    "https://mohamed-elessawy.github.io",
    "http://localhost:3000",
    "http://localhost:8501",
]

allowed_origins = getattr(config, "ALLOWED_ORIGINS", None) or DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Serenity Mental Health Chatbot API"}


@app.get("/health")
async def health_check():
    models_loaded = all(
        [
            emotion_detection.model is not None,
            language_detection._model is not None,
            rag_service.embedder is not None,
            rag_service.qdrant is not None,
        ]
    )
    status = "healthy" if models_loaded else "degraded"
    if not models_loaded:
        logger.warning("Health check: some models not loaded")
    return {"status": status, "models_loaded": models_loaded}


app.include_router(router)
