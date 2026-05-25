from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from deployment import emotion_detection
from deployment import intent_classifier
from deployment import language_detection


env_path = Path(__file__).parent / '.env'

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(dotenv_path=env_path)
    
    intent_classifier.init_client()
    
    print("Loading Emotion Detection Model...")
    emotion_detection.load()
    
    print("Loading Language Detection Model...")
    language_detection.load()
    
    print("All models loaded successfully!")

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

class TextInput(BaseModel):
    text: str

@app.post("/detect-language")
async def detect_language_endpoint(input_data: TextInput):
    language = language_detection.detect_language(input_data.text)
    return {"language": language}

@app.post("/detect-emotion")
async def detect_emotion_endpoint(input_data: TextInput):
    emotion = emotion_detection.predict(input_data.text)
    return {"emotion": emotion}

@app.post("/classify-intent")
async def classify_intent_endpoint(input_data: TextInput):
    intent = intent_classifier.classify_intent(input_data.text)
    return {"intent": intent}
