"""
Tests for Pydantic schema validation.

These test the data validation layer directly without going
through the API. Faster and more targeted than endpoint tests
for catching validation edge cases.
"""

import pytest
from pydantic import ValidationError

from deployment.schemas.chat import ChatRequest, FeedbackRequest

# ---- ChatRequest ----


def test_chat_request_valid():
    req = ChatRequest(message="Hello")
    assert req.message == "Hello"
    assert req.history == []


def test_chat_request_with_history():
    req = ChatRequest(
        message="Help me",
        history=[{"role": "user", "content": "Hi"}],
    )
    assert len(req.history) == 1


def test_chat_request_empty_string_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_whitespace_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_missing_message():
    with pytest.raises(ValidationError):
        ChatRequest()


# ---- FeedbackRequest ----


def test_feedback_valid_up():
    req = FeedbackRequest(vote="up", user_message="test", bot_response="reply")
    assert req.vote == "up"


def test_feedback_valid_down():
    req = FeedbackRequest(vote="down", user_message="test", bot_response="reply")
    assert req.vote == "down"


def test_feedback_invalid_vote():
    with pytest.raises(ValidationError):
        FeedbackRequest(vote="maybe", user_message="test", bot_response="reply")


def test_feedback_missing_fields():
    with pytest.raises(ValidationError):
        FeedbackRequest(vote="up")
