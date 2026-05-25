def get_intent_classification_prompt(user_message: str) -> str:
    """Builds the prompt used for classifying the user's intent."""

    return f"""You are an intent classification system for a mental health chatbot.

Classify the following user message into EXACTLY one of these intents:
- greeting: the user is saying hello or starting a conversation
- goodbye: the user is ending the conversation
- gratitude: the user is saying thank you or expressing appreciation
- asking_mental_health_question: the user is asking about or describing a mental health issue, emotion, or personal struggle
- out_of_scope: the message has nothing to do with mental health or conversation

Rules:
- Reply with ONLY the intent label, nothing else
- No punctuation, no explanation, just the label
- If unsure between two, pick the most likely one

User message: "{user_message}"

Intent:"""


def get_generation_system_prompt(language: str, emotion: str, intent: str, context_text: str = None) -> str:
    """Builds the dynamic system prompt based on user analysis and optional RAG context."""
    
    prompt = f"""You are a compassionate, empathetic mental health chatbot.
Your primary role is to support the user.
The user is speaking in {language}.
The user has been classified with the following dominant emotion: {emotion}.
The intent behind their message is: {intent}.
"""
    
    if context_text:
        prompt += f"\nUse the following referenced chunks to ground your knowledge and give accurate advice if applicable:\n{context_text}\n"
        
    prompt += "\nEnsure your response is caring, non-judgmental, and naturally conversational."
    return prompt
