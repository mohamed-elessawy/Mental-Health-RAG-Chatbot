import litellm
from fastapi import APIRouter
from deployment.schemas.chat import ChatRequest, ChatResponse
from deployment.schemas.prompts import get_direct_reply_prompt, get_generation_system_prompt
from deployment.core.config import config
from deployment.services.emotion_detection import predict_emotion
from deployment.services.language_detection import detect_user_language
from deployment.services.intent_classifier import classify_user_intent
from deployment.services.rag_service import rag_answer
from deployment.services.translation import translate_to_english, translate_from_english

router = APIRouter()

NON_RAG_INTENTS = {"greeting", "goodbye", "gratitude", "out_of_scope"}


@router.post("/detect-language")
async def detect_language_endpoint(request: ChatRequest):
    return {"language": detect_user_language(request.text)}


@router.post("/detect-emotion")
async def detect_emotion_endpoint(request: ChatRequest):
    return {"emotion": predict_emotion(request.text)}


@router.post("/classify-intent")
async def classify_intent_endpoint(request: ChatRequest):
    return {"intent": classify_user_intent(request.text)}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.text
    history      = [msg.model_dump() for msg in request.history]

    # Step 1 — language detection
    language = detect_user_language(user_message)

    # Step 2 — intent classification
    intent = classify_user_intent(user_message)

    # Step 3 — blocked topic: hardcoded reply
    if intent == "blocked_topic":
        return ChatResponse(
            intent=intent,
            emotion="neutral",
            language=language,
            response="I do not process or answer questions related to this topic.",
            retrieved_documents=False
        )

    # Step 4 — non-RAG intents: LLM replies naturally, no RAG, no translation
    if intent in NON_RAG_INTENTS:
        system_prompt = get_generation_system_prompt(
            emotion="neutral",
            personal_context="none",
            context=""
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = litellm.completion(
            model=config.GENERATION_LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return ChatResponse(
            intent=intent,
            emotion="neutral",
            language=language,
            response=response.choices[0].message.content.strip(),
            retrieved_documents=False
        )

    # Step 5 — mental health question: translate to English if needed
    english_message = translate_to_english(user_message, language)

    # Step 6 — emotion detection on English text
    emotion = predict_emotion(english_message)

    # Step 7 — RAG pipeline
    result = rag_answer(
        user_message=english_message,
        emotion=emotion,
        history=history
    )

    # Step 8 — translate response back if needed
    final_response = translate_from_english(result["answer"], language)

    return ChatResponse(
        intent=intent,
        emotion=emotion,
        language=language,
        response=final_response,
        retrieved_documents=True
    )