import logging

import litellm
from fastapi import APIRouter

from deployment.core.config import config
from deployment.schemas.chat import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from deployment.schemas.prompts import get_generation_system_prompt
from deployment.services.emotion_detection import predict_emotion
from deployment.services.intent_classifier import classify_user_intent
from deployment.services.language_detection import detect_user_language
from deployment.services.rag_service import rag_answer
from deployment.services.translation import (
    translate_from_english,
    translate_to_english,
)

router = APIRouter()
logger = logging.getLogger("serenity.routes")

NON_RAG_INTENTS = {"greeting", "goodbye", "gratitude", "out_of_scope", "follow_up"}


@router.post("/detect-language")
async def detect_language_endpoint(request: ChatRequest):
    return {"language": detect_user_language(request.message)}


@router.post("/detect-emotion")
async def detect_emotion_endpoint(request: ChatRequest):
    return {"emotion": predict_emotion(request.message)}


@router.post("/classify-intent")
async def classify_intent_endpoint(request: ChatRequest):
    return {"intent": classify_user_intent(request.message)}


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(request: FeedbackRequest):
    logger.info(
        "Feedback: vote=%s, user_message_length=%d",
        request.vote,
        len(request.user_message),
    )
    return FeedbackResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    history = [msg.model_dump() for msg in request.history]

    logger.info(
        "Chat request: length=%d, history_size=%d", len(user_message), len(history)
    )

    # Step 1 - language detection (sticky language for short messages)
    if len(user_message.split()) <= 3 and history:
        # Search backwards for the most recent message that has a language attached
        last_lang = next(
            (msg.get("language") for msg in reversed(history) if msg.get("language")),
            None,
        )
        language = last_lang if last_lang else detect_user_language(user_message)
    else:
        language = detect_user_language(user_message)

    logger.info("Language: %s", language)

    # Step 2 - intent classification
    intent = classify_user_intent(user_message, history)
    logger.info("Intent: %s", intent)

    # Step 3 - non-RAG intents: LLM replies directly
    if intent in NON_RAG_INTENTS:
        system_prompt = get_generation_system_prompt(
            emotion="neutral", personal_context="none", context=""
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            response = litellm.completion(
                model=config.GENERATION_LLM_MODEL, messages=messages, temperature=0.7
            )
        except Exception:
            logger.error("LLM call failed for non-RAG intent", exc_info=True)
            raise

        return ChatResponse(
            intent=intent,
            emotion="neutral",
            language=language,
            response=response.choices[0].message.content.strip(),
            retrieved_documents=False,
        )

    # Step 4 - translate to English if needed
    english_message = translate_to_english(user_message, language)

    # Step 5 - emotion detection on English text
    emotion = predict_emotion(english_message)
    logger.info("Emotion: %s", emotion)

    # Step 6 - RAG pipeline
    try:
        result = rag_answer(
            user_message=english_message, emotion=emotion, history=history
        )
    except Exception:
        logger.error("RAG pipeline failed", exc_info=True)
        raise

    # Step 7 - translate response back if needed
    final_response = translate_from_english(result["answer"], language)

    logger.info("Response generated via RAG, returning to user")

    return ChatResponse(
        intent=intent,
        emotion=emotion,
        language=language,
        response=final_response,
        retrieved_documents=True,
    )
