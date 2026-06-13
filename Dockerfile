FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies (cached unless pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Download custom models from Google Drive (cached, rarely changes)
RUN uv run python -c "\
import gdown, os; \
os.makedirs('models/distilbert', exist_ok=True); \
files = { \
    'models/distilbert/config.json': '1VYaXz6XgsCpTEzOCv-TTTPbgAZcejIiU', \
    'models/distilbert/model.safetensors': '1i0rBjiRCjTYHSKSIJH1mS-Ddk_g1j39L', \
    'models/distilbert/tokenizer_config.json': '139JAm4B6sjhY4MrGo6xmR3UYLVafbvWO', \
    'models/distilbert/tokenizer.json': '1PWPC9PpMpeqkgq9fUq9B2qhwEtwx9V1q', \
    'models/language_detector.joblib': '12dgGyGCMbAGWlW5-bL_NsrRS8XA3JqEg', \
}; \
[gdown.download(id=fid, output=path, quiet=False) for path, fid in files.items()]"

# Pre-download sentence-transformers embedding model (cached)
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code last (changes most often)
COPY deployment/ ./deployment/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]
