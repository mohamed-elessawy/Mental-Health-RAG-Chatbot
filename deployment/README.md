# Deployment Guide

This directory contains the production-grade backend server for the Mental Health RAG Chatbot. It implements a FastAPI application that orchestrates the 4-module NLP pipeline and exposes HTTP endpoints for inference.

---

## Quick Setup

**Prerequisites:** Python 3.11+ and pip installed.

### 1. Create Virtual Environment

From the **root directory** of the project:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the template and edit with your API keys:
```bash
cp deployment/.env.example deployment/.env
```

Edit `deployment/.env` with:
```dotenv
GROQ_API_KEY=gsk_your_actual_key_here
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-cluster-id.region-0.aws.cloud.qdrant.io

INTENT_LLM_MODEL="groq/llama-3.3-70b-versatile"
GENERATION_LLM_MODEL="groq/llama-3.3-70b-versatile"

RETRIEVAL_TOP_K=3
EMBED_MODEL="all-MiniLM-L6-v2"
```

---

## Running the System

### Backend (FastAPI)

From the **root directory**:
```bash
python -m uvicorn deployment.main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
Loading Emotion Detection Model...
Loading Language Detection Model...
Initializing RAG (SentenceTransformers & Qdrant)...
All backend services loaded successfully!
```

**First run takes ~5 minutes** (downloads models from Google Drive).

### Frontend (Streamlit)

In a new terminal from the **root directory**:
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Start typing in the chat box.

---

## Directory Structure

```
📁 deployment
├── 📁 api
│   └── routes.py
├── 📁 core
│   └── config.py
├── 📁 schemas
│   ├── chat.py
│   └── prompts.py
├── 📁 services
│   ├── emotion_detection.py
│   ├── intent_classifier.py
│   ├── language_detection.py
│   ├── rag_service.py
│   └── translation.py
├── .env.example
├── README.md
└── main.py
```

### File & Folder Descriptions

| Item | Purpose |
|------|---------|
| **main.py** | FastAPI entry point. Initializes the app, loads all models on startup, and includes the router. |
| **api/routes.py** | HTTP endpoints: `/detect-language`, `/detect-emotion`, `/classify-intent`, `/chat` (main orchestrator). |
| **core/config.py** | Configuration management via Pydantic `BaseSettings`. Loads `.env` variables. |
| **schemas/chat.py** | Pydantic models: `Message`, `ChatRequest`, `ChatResponse` for type safety. |
| **schemas/prompts.py** | Prompt template functions for intent classification, query rewriting, translation, response generation. |
| **services/language_detection.py** | Language detection service. Auto-downloads scikit-learn model from Google Drive. Exports `detect_user_language()`, `load_language_model()`. |
| **services/emotion_detection.py** | Emotion classification service. Auto-downloads DistilBERT model from Google Drive. Exports `predict_emotion()`, `load_emotion_model()`. |
| **services/intent_classifier.py** | Intent classification via Groq LLM. Exports `classify_user_intent()`. |
| **services/rag_service.py** | RAG orchestration. Manages SentenceTransformers embeddings and Qdrant vector DB. Exports `init_rag()`, `retrieve_documents()`, `rag_answer()`. |
| **services/translation.py** | Bidirectional translation via Groq LLM. Exports `translate_to_english()`, `translate_from_english()`. |

---

## Environment Variables

Required keys in `.env`:

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Get free API key at [console.groq.com](https://console.groq.com) |
| `QDRANT_API_KEY` | Get from Qdrant Cloud or set for local deployment |
| `QDRANT_URL` | Qdrant cluster endpoint (cloud or `http://localhost:6333` for local) |
| `INTENT_LLM_MODEL` | LLM for intent classification and query rewriting |
| `GENERATION_LLM_MODEL` | LLM for response generation |
| `RETRIEVAL_TOP_K` | Number of documents to retrieve (default: 3) |
| `EMBED_MODEL` | Embedding model name (default: `all-MiniLM-L6-v2`) |

---

## API Endpoints

**Interactive API Explorer:** Once the backend is running, visit `http://127.0.0.1:8000/docs` to explore and test all endpoints in Swagger UI.

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Detect Language
**POST** `/detect-language`
```json
{
  "text": "Bonjour, comment allez-vous?",
  "history": []
}
```
Response: `{"language": "fr"}`

### Detect Emotion
**POST** `/detect-emotion`
```json
{
  "text": "I'm feeling really overwhelmed and anxious today.",
  "history": []
}
```
Response: `{"emotion": "fear"}`

### Classify Intent
**POST** `/classify-intent`
```json
{
  "text": "What can I do about my depression?",
  "history": []
}
```
Response: `{"intent": "asking_mental_health_question"}`

### Chat (Main Endpoint)
**POST** `/chat`
```json
{
  "text": "I've been feeling very sad lately. What should I do?",
  "history": []
}
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

