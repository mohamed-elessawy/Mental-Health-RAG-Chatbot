from pathlib import Path
import joblib
import subprocess
import sys

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "language_detector.joblib"

MODEL_DRIVE_ID = "12dgGyGCMbAGWlW5-bL_NsrRS8XA3JqEg" 

_model = None

def _ensure_model():
    """Download the joblib model from Google Drive if not already present."""
    if MODEL_PATH.is_file():
        return

    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
        import gdown

    # Make sure the models folder exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Language Detection model...")
    gdown.download(id=MODEL_DRIVE_ID, output=str(MODEL_PATH), quiet=False)
    print("Language model ready.")

def load():
    """Loads the model into memory, downloading it first if necessary."""
    global _model
    _ensure_model()
    _model = joblib.load(MODEL_PATH)

def detect_language(text: str) -> str:
    """Predicts the language of the given text."""
    if _model is None:
        load() # Auto-load on first prediction
    result = _model.predict([text])
    return result[0]