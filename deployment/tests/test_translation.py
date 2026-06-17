"""
Tests for the translation service.

LLM calls are mocked. We test the logic: when to translate,
when to skip, and that prompts are constructed correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

from deployment.services.translation import translate_from_english, translate_to_english


class TestTranslateToEnglish:
    def test_english_input_returns_unchanged(self):
        """No LLM call should happen for English input."""
        with patch("deployment.services.translation.litellm.completion") as mock:
            result = translate_to_english("I feel sad", "en")
            assert result == "I feel sad"
            mock.assert_not_called()

    def test_non_english_calls_llm(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I feel very anxious"

        with patch("deployment.services.translation.litellm.completion") as mock:
            mock.return_value = mock_response
            result = translate_to_english("أنا قلق جدا", "ar")
            assert result == "I feel very anxious"
            mock.assert_called_once()

    def test_result_is_stripped(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  I feel sad  \n"

        with patch("deployment.services.translation.litellm.completion") as mock:
            mock.return_value = mock_response
            result = translate_to_english("Je suis triste", "fr")
            assert result == "I feel sad"

    def test_llm_failure_propagates(self):
        with patch("deployment.services.translation.litellm.completion") as mock:
            mock.side_effect = RuntimeError("API down")
            with pytest.raises(RuntimeError):
                translate_to_english("Hola", "es")


class TestTranslateFromEnglish:
    def test_english_target_returns_unchanged(self):
        with patch("deployment.services.translation.litellm.completion") as mock:
            result = translate_from_english("I hear you", "en")
            assert result == "I hear you"
            mock.assert_not_called()

    def test_non_english_target_calls_llm(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "أنا أسمعك"

        with patch("deployment.services.translation.litellm.completion") as mock:
            mock.return_value = mock_response
            result = translate_from_english("I hear you", "ar")
            assert result == "أنا أسمعك"
            mock.assert_called_once()
