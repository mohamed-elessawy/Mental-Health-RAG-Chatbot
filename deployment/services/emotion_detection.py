import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "distilbert"
label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

MODEL_FILES = {
    "config.json": "1VYaXz6XgsCpTEzOCv-TTTPbgAZcejIiU",
    "model.safetensors": "1i0rBjiRCjTYHSKSIJH1mS-Ddk_g1j39L",
    "tokenizer_config.json": "139JAm4B6sjhY4MrGo6xmR3UYLVafbvWO",
    "tokenizer.json": "1PWPC9PpMpeqkgq9fUq9B2qhwEtwx9V1q",
}


def _ensure_model():
    """Download model files from Google Drive if not already present."""
    if all((MODEL_PATH / f).exists() for f in MODEL_FILES):
        return

    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    for filename, file_id in MODEL_FILES.items():
        dest = MODEL_PATH / filename
        if not dest.exists():
            print(f"Downloading {filename} ...")
            gdown.download(id=file_id, output=str(dest), quiet=False)
    print("Model ready.")


tokenizer = None
model = None


def load_emotion_model():
    """Load the emotion detection model and tokenizer into memory."""
    global tokenizer, model
    _ensure_model()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()


def predict_emotion(text: str) -> str:
    """Return the predicted emotion label for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = model(**inputs).logits
    return label_names[logits.argmax().item()]


def predict_batch(texts: list[str]) -> list[str]:
    """Return predicted emotion labels for a list of texts."""
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=128, padding=True
    )
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = model(**inputs).logits
    indices = logits.argmax(dim=-1).tolist()
    return [label_names[i] for i in indices]
