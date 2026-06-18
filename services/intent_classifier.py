import litellm

from core.config import config
from schemas.prompts import get_intent_classification_prompt


def classify_user_intent(user_message: str, history: list | None = None) -> str:
    assistant_last_message = ""
    if history:
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                assistant_last_message = msg.get("content", "")
                break

    prompt = get_intent_classification_prompt(user_message, assistant_last_message)
    response = litellm.completion(
        model=config.INTENT_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()
