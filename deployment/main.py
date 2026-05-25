from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path

# Load env variables at the top before core configuration is imported
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from deployment.services import emotion_detection
from deployment.services import language_detection
from deployment.services import rag_service
from deployment.api.routes import router

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

app = FastAPI(
    title="Mental Health RAG Chatbot API",
    lifespan=lifespan
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Mental Health RAG Chatbot API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(router)
