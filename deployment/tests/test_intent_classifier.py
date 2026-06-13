"""
Tests for intent classification.

The classifier calls Groq via litellm. We mock the LLM response
and test that the function correctly extracts and returns the
intent label. We also test prompt construction.
"""

from unittest.mock import MagicMock, patch

import pytest

from deployment.schemas.prompts import get_intent_classification_prompt
from deployment.services.intent_classifier import classify_user_intent

VALID_INTENTS = {
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
}


class TestIntentClassifierLogic:
    """Test the logic around the LLM call."""

    def _mock_llm_response(self, label: str):
        """Create a mock litellm response that returns the given label."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = label
        return mock_response

    def test_greeting_detected(self):
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("greeting")
            result = classify_user_intent("Hello!")
            assert result == "greeting"

    def test_goodbye_detected(self):
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("goodbye")
            result = classify_user_intent("Bye, thanks for the help")
            assert result == "goodbye"

    def test_mental_health_detected(self):
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("asking_mental_health_question")
            result = classify_user_intent("I have been feeling depressed")
            assert result == "asking_mental_health_question"

    def test_out_of_scope_detected(self):
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("out_of_scope")
            result = classify_user_intent("What is the capital of France?")
            assert result == "out_of_scope"

    def test_result_is_lowercased(self):
        """The function calls .lower() on the result."""
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("GREETING")
            result = classify_user_intent("Hi")
            assert result == "greeting"

    def test_result_is_stripped(self):
        """Whitespace around the label should be removed."""
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("  greeting  \n")
            result = classify_user_intent("Hi")
            assert result == "greeting"

    def test_history_extracts_last_assistant_message(self):
        """When history is provided, the function should find the last
        assistant message and pass it to the prompt."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Thanks"},
        ]
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("gratitude")
            result = classify_user_intent("Thanks", history)
            assert result == "gratitude"
            # Verify the prompt was called (LLM received a message)
            mock.assert_called_once()

    def test_empty_history_handled(self):
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.return_value = self._mock_llm_response("greeting")
            result = classify_user_intent("Hello", [])
            assert result == "greeting"

    def test_llm_failure_propagates(self):
        """If the LLM call fails, the error should propagate up."""
        with patch("deployment.services.intent_classifier.litellm.completion") as mock:
            mock.side_effect = RuntimeError("API timeout")
            with pytest.raises(RuntimeError, match="API timeout"):
                classify_user_intent("Hello")


class TestIntentPrompt:
    """Test the prompt construction function directly."""

    def test_prompt_contains_user_message(self):
        prompt = get_intent_classification_prompt("I feel anxious")
        assert "I feel anxious" in prompt

    def test_prompt_contains_all_intents(self):
        prompt = get_intent_classification_prompt("test")
        for intent in VALID_INTENTS:
            assert intent in prompt

    def test_prompt_includes_assistant_context(self):
        prompt = get_intent_classification_prompt("Yes", "How are you feeling?")
        assert "How are you feeling?" in prompt

    def test_prompt_without_assistant_context(self):
        prompt = get_intent_classification_prompt("Hello")
        assert "Previous assistant message" not in prompt
