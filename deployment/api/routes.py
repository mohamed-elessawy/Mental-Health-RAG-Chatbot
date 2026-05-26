import litellm
import os
from fastapi import APIRouter

from deployment.schemas.chat import ChatRequest, ChatResponse
from deployment.core.config import config
from deployment.schemas.prompts import get_generation_system_prompt
from deployment.services.emotion_detection import predict_emotion
from deployment.services.language_detection import detect_user_language
from deployment.services.intent_classifier import classify_user_intent
from deployment.services.rag_service import retrieve_documents

router = APIRouter()

@router.post("/detect-language")
async def detect_language_endpoint(request: ChatRequest):
    language = detect_user_language(request.text)
    return {"language": language}

@router.post("/detect-emotion")
async def detect_emotion_endpoint(request: ChatRequest):
    emotion = predict_emotion(request.text)
    return {"emotion": emotion}

@router.post("/classify-intent")
async def classify_intent_endpoint(request: ChatRequest):
    intent = classify_user_intent(request.text)
    return {"intent": intent}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.text
    
    # Run preliminary checks
    language = detect_user_language(user_message)
    emotion = predict_emotion(user_message)
    intent = classify_user_intent(user_message)
    
    retrieved_documents = False
    context_text = ""
    
    # Only search on Qdrant data if the user is asking for mental health advice
    if intent == "asking_mental_health_question":
        try:
            docs = retrieve_documents(query=user_message)
            context_text = "Retrieved Context:\n" + "\n\n".join(docs)
            retrieved_documents = True
        except Exception as e:

            print(f"Warning: RAG retrieval failed. {e}")
            context_text = None
            

    system_prompt = get_generation_system_prompt(
        language=language, 
        emotion=emotion, 
        intent=intent, 
        context_text=context_text if retrieved_documents else None
    )

    response = litellm.completion(
        model=config.GENERATION_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    
    generated_text = response.choices[0].message.content.strip()
    
    return ChatResponse(
        intent=intent,
        emotion=emotion,
        language=language,
        response=generated_text,
        retrieved_documents=retrieved_documents
    )
