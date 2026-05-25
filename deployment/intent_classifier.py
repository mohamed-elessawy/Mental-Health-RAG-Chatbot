import os
from groq import Groq

client = None

def init_client():
    global client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def classify_intent(user_message: str) -> str:
    prompt = f"""You are an intent classification system for a mental health chatbot.

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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20
    )
    return response.choices[0].message.content.strip().lower()