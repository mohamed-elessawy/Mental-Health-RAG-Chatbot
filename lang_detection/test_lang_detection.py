from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import joblib
from sklearn.pipeline import Pipeline

# Paths relative to this file (lang_detection/), not the repo root.
LANG_DETECTION_ROOT = Path(__file__).resolve().parent
MODEL_PATH_DEFAULT = LANG_DETECTION_ROOT / "models" / "language_detector.joblib"
METRICS_DIR = LANG_DETECTION_ROOT / "metrics"
@dataclass
class TrainResult:
    model: Pipeline
    test_accuracy: float
    report: str
    confusion_matrix_path: Path
    confusion_matrix_png_path: Path


def _as_list(x: Iterable[str]) -> List[str]:
    return list(x)


def load(model_path: Union[str, Path, None] = None) -> Pipeline:
    path = Path(model_path) if model_path is not None else MODEL_PATH_DEFAULT
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    model: Pipeline = joblib.load(path)

    try:
        features = model.named_steps.get("features")
        if hasattr(features, "n_jobs"):
            setattr(features, "n_jobs", 1)
    except Exception:
        pass
    return model


def predict(model: Pipeline, texts: Sequence[str]) -> List[str]:
    return list(model.predict(list(texts)))


if __name__ == "__main__":
    model = load()
    print("Model loaded! (sklearn)")

    texts = [
        "How are you?",
        "كيف حالك",
        "Bonjour mon ami",
        "Guten Morgen",
        "Hola amigo",
        "La tecnologia dell'intelligenza artificiale sta trasformando rapidamente il modo in cui le persone lavorano e comunicano.",
        "Durante nuestras vacaciones visitamos varias ciudades históricas, probamos comida tradicional y conocimos a personas muy amables.",
    ]

    predictions = predict(model, texts)

    for text, lang in zip(texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted Language: {lang}")
        print("-" * 30)