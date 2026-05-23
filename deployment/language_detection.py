from pathlib import Path
from typing import List, Sequence, Union
import joblib
from sklearn.pipeline import Pipeline

# Points to models/ folder at repo root
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "language_detector.joblib"

_model = None

def load():
    global _model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Download it from the Drive link in README and place it in models/"
        )
    _model = joblib.load(MODEL_PATH)

def detect_language(text: str) -> str:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load() first.")
    result = _model.predict([text])
    return result[0]