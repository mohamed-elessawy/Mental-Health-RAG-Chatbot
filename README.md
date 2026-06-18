# Serenity: Mental Health Support Chatbot (Backend)

Production backend for **Serenity**, a mental health support chatbot. It is a FastAPI
application that orchestrates a multi-stage NLP pipeline (language detection → intent
classification → translation → emotion detection → RAG retrieval → empathetic response
generation) and exposes HTTP endpoints consumed by the provided chat frontend.

---

## Deliverables

| Item | Link |
|------|------|
| Backend repo (this repo, `mlops` branch) | https://github.com/mohamed-elessawy/Mental-Health-RAG-Chatbot/tree/mlops |
| Deployed API (Hugging Face Space) | https://alaasrour-serenity-backend.hf.space |
| Forked frontend repo | https://github.com/mohamed-elessawy/chatbot-frontend |
| Deployed frontend (GitHub Pages) | https://mohamed-elessawy.github.io/chatbot-frontend/ |
| Docker layer cache verification | [cached_build.png](https://raw.githubusercontent.com/mohamed-elessawy/Mental-Health-RAG-Chatbot/mlops/assets/cached_build.png) |
| Monitoring dashboard screenshot | [dashboard.png](https://raw.githubusercontent.com/mohamed-elessawy/Mental-Health-RAG-Chatbot/mlops/assets/dashboard.png) |

### A note on the `mlops` branch

This project lives on the **`mlops` branch**, and CI/CD runs on pushes to `mlops`.

The `main` branch holds the full original NLP project: training notebooks, datasets,
model artifacts, and experiment outputs. Those folders are not appropriate to ship as an
MLOps production backend (large, non-deployable, and noisy for a grader). We developed
and merged everything on `main` while building the project, then later carved out this
clean, deployment-only backend onto the `mlops` branch. **Treat `mlops` as the
deliverable branch for this submission.**

---

## NLP Approach & Design Decisions

The `/chat` request flows through a sequence of focused stages, each isolated in its own
service module under [services/](services/):

1. **Language detection**: a TF-IDF + LinearSVC scikit-learn model
   ([services/language_detection.py](services/language_detection.py)). For very short
   messages (≤3 words) we *stick* to the last detected language from history, since short
   inputs like "ok" are unreliable to classify.
2. **Intent classification**: zero-shot LLM prompting via Groq
   ([services/intent_classifier.py](services/intent_classifier.py)), temperature 0 for
   determinism. Labels: `greeting`, `goodbye`, `gratitude`,
   `asking_mental_health_question`, `out_of_scope`. Any unrecognized label is mapped to
   `out_of_scope` so the bot fails safe.
3. **Non-RAG intents** (greeting / goodbye / gratitude / out-of-scope) get a short,
   on-brand reply from a system prompt, with no retrieval and no wasted latency.
4. **Translation to English** (skipped for English input) so retrieval and generation run
   in one consistent language ([services/translation.py](services/translation.py)).
5. **Emotion detection**: a fine-tuned DistilBERT 6-class classifier (sadness, joy, love,
   anger, fear, surprise) so the response can be emotion-aware
   ([services/emotion_detection.py](services/emotion_detection.py)).
6. **RAG** ([services/rag_service.py](services/rag_service.py)):
   - *Query rewriting* extracts personal context and a focused search query.
   - *Retrieval* runs a dense vector search over a mental-health Q&A knowledge base in
     **Qdrant**, embedded with `all-MiniLM-L6-v2` (sentence-transformers).
   - *Generation* feeds retrieved references + detected emotion + personal context + recent
     history to the generation LLM, with explicit instructions to recommend professional
     help if the user appears to be in crisis.
7. **Translation back** to the user's language (skipped for English).

**Why this design:** intent classification gates the expensive RAG path so greetings and
out-of-scope messages stay cheap; retrieval grounds answers in real counselor responses
rather than letting the LLM hallucinate; and emotion + personal context make replies feel
empathetic rather than generic.

**LLM provider:** Groq (via `litellm`), for fast, free-tier-friendly inference.
**Vector DB:** Qdrant Cloud.

### Edge cases handled

- **Empty / whitespace messages** → rejected with `422` via a Pydantic validator
  ([schemas/chat.py](schemas/chat.py)).
- **Out-of-scope questions** → classified as `out_of_scope`, answered with a polite
  boundary-setting reply instead of retrieval.
- **Crisis / safety** → the generation system prompt instructs the model to surface
  professional-help guidance when the user appears to be in distress.
- **Invalid feedback votes** → only `up` / `down` accepted, else `422`.

---

## Repository Structure

```
📦 Mental-Health-RAG-Chatbot/   (mlops branch)
├── 📁 .github/workflows/
│   └── ci.yml                  # Lint → test → build/push image → deploy to HF
├── 📁 assets/
│   ├── cached_build.png        # Docker layer cache hit verification
│   └── dashboard.png           # Axiom monitoring dashboard
├── 📁 api/
│   └── routes.py               # HTTP endpoints
├── 📁 core/
│   ├── config.py               # Pydantic settings (loaded from .env)
│   └── logging.py              # Centralized "serenity" logger setup
├── 📁 monitoring/
│   ├── app_metrics.py          # Custom NLP / data metrics
│   ├── config.py               # OpenTelemetry settings
│   ├── instrument.py           # Wires OTel into the FastAPI app
│   ├── middleware.py           # HTTP server metrics middleware
│   ├── otel.py                 # OTLP exporter setup (traces/metrics/logs)
│   └── axiom-dashboard-queries.txt
├── 📁 schemas/
│   ├── chat.py                 # Request/response models + validators
│   └── prompts.py              # LLM prompt templates
├── 📁 services/                # NLP pipeline (no HTTP dependencies)
│   ├── language_detection.py
│   ├── emotion_detection.py
│   ├── intent_classifier.py
│   ├── rag_service.py
│   └── translation.py
├── 📁 tests/                   # pytest suite (slow model tests marked @pytest.mark.slow)
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures (mocked client, real-model loaders)
│   ├── test_emotion_detection.py
│   ├── test_endpoints.py
│   ├── test_intent_classifier.py
│   ├── test_language_detection.py
│   ├── test_rag_service.py
│   ├── test_schemas.py
│   └── test_translation.py
├── main.py                     # FastAPI entry point, CORS, lifespan, monitoring
├── .env.example
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

> Model files (`models/`) are **not** committed. They are fetched from Google Drive at
> Docker build time (and on first local run), so the repo stays light.

---

## Quick Setup

Prerequisites: **Python 3.12+** and **uv** installed.

### 1. Install dependencies

```bash
uv sync            # runtime only
uv sync --dev      # + test and lint tooling
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GROQ_API_KEY=gsk_your_key_here
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-cluster-id.region-0.aws.cloud.qdrant.io

INTENT_LLM_MODEL=groq/llama-3.1-8b-instant
GENERATION_LLM_MODEL=groq/openai/gpt-oss-120b

RETRIEVAL_TOP_K=3
EMBED_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
ALLOWED_ORIGINS=["https://mohamed-elessawy.github.io","http://localhost:8501"]
```

---

## Running the System

```bash
uv run uvicorn main:app --reload
```

The API serves on `http://127.0.0.1:8000` (interactive docs at `/docs`). First run takes a
few minutes while models download from Google Drive.

---

## Running with Docker

The `Dockerfile` installs CPU-only PyTorch and bakes the language detector and emotion
model into the image at build time, so no Google Drive download is needed at container
startup. The container listens on port **7860** (the Hugging Face Spaces convention).

```bash
docker build -t mental-health-chatbot .
docker run --env-file .env -p 8000:7860 mental-health-chatbot
```

The API is then available at `http://localhost:8000`.

### Layer caching

The Dockerfile is ordered so the expensive steps come first and the application code is
copied last. Dependencies (`uv sync`), the Google Drive model download, and the
sentence-transformers embedding download all live above the `COPY` of the app source.
Because of this, a code-only change reuses the cached layers and only the final copy step
rebuilds. In practice **7 of the 8 build steps are cache hits** on a code change, as shown
below:

![Docker Layer Cache Verification](assets\cached_build.png)

---

## Running Tests

The suite covers endpoints, core services, and validation logic with `pytest` /
`pytest-cov`.

```bash
uv run pytest -v -m "not slow"                              # fast, CI-equivalent
uv run pytest -v                                            # all tests incl. real models
uv run pytest --cov=. --cov-report=term-missing -v          # with coverage
```

Latest full local run (including live models): **79 passed, ~95% coverage** (800
statements, 39 missed). Slow tests load real models and are excluded from CI to keep the
pipeline fast.

---

## Pre-commit Hooks

Linting and formatting run automatically on every commit:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## CI/CD Pipeline

Defined in [.github/workflows/ci.yml](.github/workflows/ci.yml) and runs on every push to
the **`mlops`** branch (see the branch note above). The Docker build step reuses the
GitHub Actions layer cache (see [Layer caching](#layer-caching) above).

```
push to mlops
     |
     v
1. Lint & Unit Tests   (ruff check + ruff format --check + pytest -m "not slow")
     |  fails -> pipeline stops
     v
2. Build & Push Docker Image
     |  linux/amd64, GitHub Actions layer cache, pushed to Docker Hub (<sha> + latest)
     v
3. Deploy to Hugging Face Spaces
     |  force-push the repo -> HF rebuilds the Space container
     v
Live at https://alaasrour-serenity-backend.hf.space
```

### Required GitHub Secrets & Variables

**Secrets:** `DOCKERHUB_TOKEN`, `HF_TOKEN`
**Variables:** `DOCKERHUB_USERNAME`, `HF_USERNAME`, `HF_SPACE`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key. Used by litellm for all LLM calls. |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key. |
| `QDRANT_URL` | Yes | Qdrant cluster endpoint. |
| `INTENT_LLM_MODEL` | No | LLM for intent classification + query rewriting. Default: `groq/llama-3.1-8b-instant` |
| `GENERATION_LLM_MODEL` | No | LLM for response generation. Default: `groq/openai/gpt-oss-120b` |
| `RETRIEVAL_TOP_K` | No | Number of documents to retrieve. Default: `3` |
| `EMBED_MODEL` | No | Sentence embedding model. Default: `all-MiniLM-L6-v2` |
| `LOG_LEVEL` | No | Logging verbosity. Default: `INFO` |
| `ALLOWED_ORIGINS` | No | CORS allowed origins (JSON list). Default: GitHub Pages frontend + localhost |

---

## API Endpoints

Interactive explorer at `http://127.0.0.1:8000/docs` when the backend is running.

### GET /health

```json
{"status": "healthy", "models_loaded": true}
```

### POST /chat

Request:
```json
{ "message": "I have been feeling very sad lately. What should I do?", "history": [] }
```

Response:
```json
{
  "intent": "asking_mental_health_question",
  "emotion": "sadness",
  "language": "en",
  "response": "I hear you. Sadness is a natural emotion...",
  "retrieved_documents": true
}
```

### POST /feedback

Request:
```json
{ "vote": "up", "user_message": "I feel anxious", "bot_response": "I hear you..." }
```

Response:
```json
{"status": "ok"}
```

### Utility endpoints

- `POST /detect-language` → `{"language": "fr"}`
- `POST /detect-emotion` → `{"emotion": "fear"}`
- `POST /classify-intent` → `{"intent": "asking_mental_health_question"}`

---

## Monitoring

The API is instrumented with **OpenTelemetry** and exports traces, metrics, and logs
directly to [Axiom](https://axiom.co) over OTLP/HTTP. No separate Collector container is
required.

### Enable monitoring

Add to `.env`:

```dotenv
OTEL_ENABLED=true
AXIOM_TOKEN=your_axiom_api_token
OTEL_SERVICE_NAME=serenity-mental-health-api
OTEL_SERVICE_VERSION=0.1.0
OTEL_DEPLOYMENT_ENVIRONMENT=development
```

Then run the API normally (`uv run uvicorn main:app`); instrumentation is wired into
`main.py` and activates from the env vars above.

### The three chosen metrics

These are the three required metrics, one per layer:

| # | Category | Metric | Recorded when | Why we track it |
|---|----------|--------|---------------|-----------------|
| 1 | **Model / NLP** | `nlp.intent.classified` | After intent classification on `/chat` | Shows the user-intent distribution. A spike in `out_of_scope` flags prompt issues or misuse, and the mix of intents tells us what users actually come to the bot for. |
| 2 | **Data** | `data.chat.message.length` | On every `/chat` request | Detects abnormal input sizes. Very short or empty messages can signal bot-like spam, while very long ones often indicate abuse or prompt-injection attempts. |
| 3 | **Server** | `http.server.request.count` | On every HTTP request (middleware) | Tracks request volume and availability. The `http.status_code` attribute lets us derive per-route error rate. |

### Additional signals

We also export these supporting metrics, with the same reasoning broken out:

| Category | Metric | Recorded when | Why we track it |
|----------|--------|---------------|-----------------|
| **Data** | `data.feedback.vote` | When the frontend posts to `/feedback` | Captures the thumbs up / down ratio, our most direct signal of whether responses are actually helpful. |
| **Server** | `http.server.request.duration` | On every HTTP request (middleware) | Latency distribution per route, used to catch slow LLM or retrieval calls degrading the user experience. |
| **Server** | `http.server.active_requests` | Incremented/decremented around each request | Live concurrency, which shows load spikes and whether the service is keeping up. |
| **Server** | `server.process.uptime` | Sampled by an observable gauge | Seconds since process start, a simple health and restart-detection signal. |

### Axiom dashboard

![Axiom metrics dashboard](assets\dashboard.png)

| Panel | Metric | Category |
|-------|--------|----------|
| Intent Classification Rate | `nlp.intent.classified` | Model / NLP |
| Message Length | `data.chat.message.length` | Data |
| Request Volume Over Time | `http.server.request.count` | Server |
| User Feedback Volume | `data.feedback.vote` | Data |
| Concurrent Requests | `http.server.active_requests` | Server |
| Process Uptime | `server.process.uptime` | Server |

Dashboard queries: [monitoring/axiom-dashboard-queries.txt](monitoring/axiom-dashboard-queries.txt)
