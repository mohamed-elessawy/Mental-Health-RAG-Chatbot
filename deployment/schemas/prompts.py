def get_intent_classification_prompt(
    user_message: str, assistant_last_message: str = ""
) -> str:
    context_str = (
        f'\nPrevious assistant message: "{assistant_last_message}"'
        if assistant_last_message
        else ""
    )
    return f"""You are an intent classification system for a mental health chatbot.

Classify the following user message into EXACTLY one of these intents:
- greeting: the user is saying hello or starting a conversation
- goodbye: the user is ending the conversation
- gratitude: the user is saying thank you or expressing appreciation
- follow_up: the user is briefly answering a question the assistant just asked
- asking_mental_health_question: the user is asking about or describing a mental health issue, emotion, or personal struggle
- out_of_scope: the message has nothing to do with mental health or conversation

Rules:
- Reply with ONLY the intent label, nothing else
- No punctuation, no explanation, just the label
- If unsure between two, pick the most likely one

Context: {context_str}

User message: "{user_message}"

Intent:"""


def get_query_rewrite_prompt(user_message: str) -> str:
    return f"""You are helping a mental health support chatbot find relevant information.

Given the user message, do two things:

1. Extract personal context: note gender, relationship, or specific situation.
   If none found, write "none".

2. Write a focused search query that:
   - Removes names and irrelevant personal details
   - Keeps the core emotional and mental health concern
   - Expands vague terms into descriptive emotional language
   - Is between 10 and 30 words

Respond in this exact format:
CONTEXT: <personal context>
QUERY: <search query>

User message: "{user_message}"
"""


def get_generation_system_prompt(
    emotion: str, personal_context: str, context: str
) -> str:
    return f"""You are a compassionate mental health support assistant.

User emotional state : {emotion}
Personal context     : {personal_context}

Guidelines:
- Use the counseling examples below as reference for tone and approach.
- Always be empathetic, non-judgmental, and supportive.
- Use personal context to personalize your response.
- Do NOT diagnose. Do NOT prescribe medication.
- If user seems in crisis, recommend professional help.
- IMPORTANT RULE: If the user asks about or mentions LGBTQ+, sexual orientation, or gender identity topics, you MUST politely refuse to answer and state that you do not process or answer questions related to this topic. Do not engage in discussion about it.

Reference counseling examples:
{context}"""


def get_translation_prompt(text: str, target_language: str) -> str:
    return f"Translate this to {target_language}. Return only the translation, nothing else.\n\n{text}"


def get_direct_reply_prompt(reply: str, language: str) -> str:
    return f"Translate this to {language}. Return only the translation.\n\n{reply}"


DIRECT_REPLIES = {
    "blocked_topic": "I do not process or answer questions related to this topic."
}
