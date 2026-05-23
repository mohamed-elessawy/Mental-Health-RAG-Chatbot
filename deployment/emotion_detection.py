from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH  = Path(__file__).parent.parent / "models" / "distilbert"
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Load once at import time
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()


def predict(text: str) -> str:
    """Return the predicted emotion label for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = model(**inputs).logits
    return label_names[logits.argmax().item()]


def predict_batch(texts: list[str]) -> list[str]:
    """Return predicted emotion labels for a list of texts."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       max_length=128, padding=True)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = model(**inputs).logits
    indices = logits.argmax(dim=-1).tolist()
    return [label_names[i] for i in indices]


if __name__ == "__main__":
    demo_texts = [
        # formal
        "I feel so happy today!",
        "I am really scared about tomorrow",
        "This makes me so angry",
        "I love spending time with you",
        "I feel so sad and empty",
        "Wow I did not see that coming!",
        # casual
        "man i just got rejected it hurts so bad",
        "bro i aced the exam lets gooo",
        "idk why but i just feel empty today",
        "she said yes!!!",
        "I cant believe they did that to me",
        "this is the best day of my life ngl",
    ]

    print("=== Emotion Detection — Smoke Test ===\n")
    for text in demo_texts:
        print(f"  '{text}' --> {predict(text)}")
