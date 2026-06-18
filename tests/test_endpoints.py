"""
Tests for all API endpoints.

Each test gets a `client` fixture from conftest.py where all
external services (Groq, Qdrant, models) are mocked. We test
that the API handles requests correctly, not that the models
produce correct predictions.
"""

from unittest.mock import patch

import litellm
import pytest

# ---- Health and root ----


def test_root_returns_welcome(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_returns_healthy_when_models_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["models_loaded"] is True


def test_health_returns_degraded_when_models_missing(client):
    """If a model failed to load, health should report degraded."""
    with patch("services.emotion_detection.model", None):
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "degraded"
        assert body["models_loaded"] is False


# ---- Feedback endpoint ----


def test_feedback_thumbs_up(client):
    response = client.post(
        "/feedback",
        json={
            "vote": "up",
            "user_message": "I feel anxious",
            "bot_response": "I hear you.",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_feedback_thumbs_down(client):
    response = client.post(
        "/feedback",
        json={
            "vote": "down",
            "user_message": "I feel anxious",
            "bot_response": "I hear you.",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_feedback_invalid_vote_rejected(client):
    """vote must be 'up' or 'down', anything else is 422."""
    response = client.post(
        "/feedback",
        json={
            "vote": "maybe",
            "user_message": "I feel anxious",
            "bot_response": "I hear you.",
        },
    )
    assert response.status_code == 422


def test_feedback_missing_fields_rejected(client):
    response = client.post("/feedback", json={"vote": "up"})
    assert response.status_code == 422


# ---- Chat endpoint: happy paths ----


def test_chat_mental_health_question(client):
    """Default mock intent is asking_mental_health_question, so RAG runs."""
    response = client.post(
        "/chat", json={"message": "I have been feeling very anxious lately"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "asking_mental_health_question"
    assert body["emotion"] == "sadness"
    assert body["language"] == "en"
    assert body["retrieved_documents"] is True
    assert len(body["response"]) > 0


def test_chat_greeting(client):
    """When intent is greeting, no RAG, direct LLM reply."""
    mock_response = litellm.ModelResponse(
        choices=[
            litellm.Choices(
                message=litellm.Message(
                    content="Hello! How are you feeling today?", role="assistant"
                )
            )
        ]
    )
    with (
        patch(
            "api.routes.classify_user_intent",
            return_value="greeting",
        ),
        patch("api.routes.litellm.completion", return_value=mock_response),
    ):
        response = client.post("/chat", json={"message": "Hello!"})
        body = response.json()
        assert body["intent"] == "greeting"
        assert body["retrieved_documents"] is False
        assert "Hello" in body["response"]


def test_chat_out_of_scope(client):
    """Out of scope messages get an LLM reply staying on topic."""
    mock_response = litellm.ModelResponse(
        choices=[
            litellm.Choices(
                message=litellm.Message(
                    content="I can only help with mental health topics.",
                    role="assistant",
                )
            )
        ]
    )
    with (
        patch(
            "api.routes.classify_user_intent",
            return_value="out_of_scope",
        ),
        patch("api.routes.litellm.completion", return_value=mock_response),
    ):
        response = client.post("/chat", json={"message": "What is the weather today?"})
        body = response.json()
        assert body["intent"] == "out_of_scope"
        assert body["retrieved_documents"] is False
        assert "mental health" in body["response"].lower()


# ---- Chat endpoint: error cases ----


def test_chat_empty_message_rejected(client):
    """Empty string should fail validation."""
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_whitespace_only_rejected(client):
    """Whitespace-only string should fail validation."""
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_missing_message_field(client):
    """No message field at all."""
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_wrong_http_method(client):
    """GET on /chat should be 405 Method Not Allowed."""
    response = client.get("/chat")
    assert response.status_code == 405


# use pytest, so the test will pass
def test_chat_rag_failure_returns_500(client):
    """If RAG pipeline throws, the error should propagate."""
    with patch(
        "api.routes.rag_answer",
        side_effect=RuntimeError("Qdrant connection lost"),
    ):
        with pytest.raises(RuntimeError, match="Qdrant connection lost"):
            client.post("/chat", json={"message": "I feel really depressed"})


def test_chat_with_history(client):
    """Verify history is accepted and does not break the flow."""
    response = client.post(
        "/chat",
        json={
            "message": "I cannot sleep at night",
            "history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello, how can I help?"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["retrieved_documents"] is True


# ---- Utility endpoints ----


def test_detect_language(client):
    response = client.post("/detect-language", json={"message": "Hello there"})
    assert response.status_code == 200
    assert response.json()["language"] == "en"


def test_detect_emotion(client):
    response = client.post("/detect-emotion", json={"message": "I am so sad"})
    assert response.status_code == 200
    assert response.json()["emotion"] == "sadness"


def test_classify_intent(client):
    response = client.post("/classify-intent", json={"message": "Hi there"})
    assert response.status_code == 200
    assert "intent" in response.json()
