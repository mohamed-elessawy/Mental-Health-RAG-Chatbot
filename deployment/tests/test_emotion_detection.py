"""
Tests for the emotion detection service.

Uses the real DistilBERT model. Marked as slow.
"""

import pytest

from deployment.services.emotion_detection import predict_batch, predict_emotion

VALID_LABELS = {"sadness", "joy", "love", "anger", "fear", "surprise"}


@pytest.mark.slow
class TestEmotionDetection:
    def test_model_loads_successfully(self, emotion_model):
        model, tokenizer = emotion_model
        assert model is not None
        assert tokenizer is not None

    def test_predict_sadness(self, emotion_model):
        result = predict_emotion("I feel so sad and hopeless")
        assert result == "sadness"

    def test_predict_joy(self, emotion_model):
        result = predict_emotion("I am so happy and grateful today")
        assert result == "joy"

    def test_predict_anger(self, emotion_model):
        result = predict_emotion("I am furious and cannot control my rage")
        assert result == "anger"

    def test_predict_fear(self, emotion_model):
        result = predict_emotion("I am terrified about what might happen")
        assert result == "fear"

    def test_returns_valid_label(self, emotion_model):
        """Output must be one of the six known labels."""
        result = predict_emotion("I do not know how I feel")
        assert result in VALID_LABELS

    def test_returns_string(self, emotion_model):
        result = predict_emotion("Hello")
        assert isinstance(result, str)

    def test_long_text_does_not_crash(self, emotion_model):
        """Model truncates at 128 tokens. Make sure long text works."""
        long_text = "I feel overwhelmed. " * 100
        result = predict_emotion(long_text)
        assert result in VALID_LABELS

    def test_batch_prediction(self, emotion_model):
        texts = ["I am happy", "I am sad", "I am angry"]
        results = predict_batch(texts)
        assert len(results) == 3
        assert all(r in VALID_LABELS for r in results)

    def test_batch_single_item(self, emotion_model):
        results = predict_batch(["I feel scared"])
        assert len(results) == 1
        assert results[0] in VALID_LABELS
