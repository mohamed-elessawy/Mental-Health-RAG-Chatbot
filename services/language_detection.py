import logging
import subprocess
import sys
from pathlib import Path

import joblib

logger = logging.getLogger("serenity.services.language")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "language_detector.joblib"
)

MODEL_DRIVE_ID = "12dgGyGCMbAGWlW5-bL_NsrRS8XA3JqEg"

_model = None


def _ensure_model():
    """Download the joblib model from Google Drive if not already present."""
    if MODEL_PATH.is_file():
        return

    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    # Make sure the models folder exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Language Detection model...")
    gdown.download(id=MODEL_DRIVE_ID, output=str(MODEL_PATH), quiet=False)
    logger.info("Language model ready.")


def load_language_model():
    """Loads the model into memory, downloading it first if necessary."""
    global _model
    _ensure_model()
    _model = joblib.load(MODEL_PATH)


def detect_user_language(text: str) -> str:
    """Predicts the language of the given text."""
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load() first before inference.")
    result = _model.predict([text])
    return result[0]
