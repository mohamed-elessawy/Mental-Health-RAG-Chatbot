import litellm
import os
from deployment.core.config import config
from deployment.schemas.prompts import get_intent_classification_prompt

def classify_user_intent(user_message: str) -> str:
    prompt = get_intent_classification_prompt(user_message)
    response = litellm.completion(
        model=config.INTENT_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200 # this number represent input/output tokens.
    )
    return response.choices[0].message.content.strip().lower()
