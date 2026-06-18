"""
Shared test fixtures.

Two modes:
- Fast tests (default): all services mocked, no models loaded.
  These run in CI.
- Slow tests (@pytest.mark.slow): load real models from disk.
  Run locally with: pytest -m slow

Model load times are printed so you know exactly how long
each one takes.
"""

import time
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---- Pytest configuration ----


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: requires real models on disk")


# ---- Fast client (mocked, for endpoint tests) ----


@asynccontextmanager
async def _empty_lifespan(app):
    yield


@pytest.fixture()
def client():
    with (
        patch("main.lifespan", _empty_lifespan),
        patch(
            "api.routes.detect_user_language",
            return_value="en",
        ),
        patch(
            "api.routes.predict_emotion",
            return_value="sadness",
        ),
        patch(
            "api.routes.classify_user_intent",
            return_value="asking_mental_health_question",
        ),
        patch(
            "api.routes.translate_to_english",
            side_effect=lambda text, lang: text,
        ),
        patch(
            "api.routes.translate_from_english",
            side_effect=lambda text, lang: text,
        ),
        patch(
            "api.routes.rag_answer",
            return_value={
                "answer": "I hear you. That sounds really difficult.",
                "search_query": "feeling anxious",
                "personal_context": "none",
                "sources": ["How to cope with anxiety"],
            },
        ),
        patch("services.emotion_detection.model", "fake-model"),
        patch("services.language_detection._model", "fake-model"),
        patch("services.rag_service.embedder", "fake-embedder"),
        patch("services.rag_service.qdrant", "fake-qdrant"),
    ):
        from main import app

        with TestClient(app) as c:
            yield c


# ---- Slow fixtures (real models, for service tests) ----


@pytest.fixture(scope="session")
def language_model():
    """Load the real language detection model once for all tests."""
    from services.language_detection import _model, load_language_model

    if _model is not None:
        return _model

    start = time.time()
    load_language_model()
    elapsed = time.time() - start
    print(f"\n[TIMING] Language model loaded in {elapsed:.2f}s")

    from services.language_detection import _model

    return _model


@pytest.fixture(scope="session")
def emotion_model():
    """Load the real emotion detection model once for all tests."""
    from services.emotion_detection import (
        load_emotion_model,
        model,
        tokenizer,
    )

    if model is not None:
        return model, tokenizer

    start = time.time()
    load_emotion_model()
    elapsed = time.time() - start
    print(f"\n[TIMING] Emotion model loaded in {elapsed:.2f}s")

    from services.emotion_detection import model, tokenizer

    return model, tokenizer
