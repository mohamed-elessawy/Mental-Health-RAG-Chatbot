import litellm
from deployment.core.config import config
from deployment.schemas.prompts import get_translation_prompt


def translate_to_english(text: str, source_language: str) -> str:
    if source_language == "en":
        return text
    response = litellm.completion(
        model=config.INTENT_LLM_MODEL,
        messages=[{
            "role": "user",
            "content": get_translation_prompt(text, "English")
        }],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def translate_from_english(text: str, target_language: str) -> str:
    if target_language == "en":
        return text
    response = litellm.completion(
        model=config.INTENT_LLM_MODEL,
        messages=[{
            "role": "user",
            "content": get_translation_prompt(text, target_language)
        }],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()