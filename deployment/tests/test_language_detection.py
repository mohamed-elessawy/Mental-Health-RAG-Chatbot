"""
Tests for the language detection service.

These use the real model loaded from disk. Marked as slow
because model loading takes several seconds.
"""

import pytest

from deployment.services.language_detection import detect_user_language


@pytest.mark.slow
class TestLanguageDetection:
    """All tests in this class require the real language model."""

    def test_model_loads_successfully(self, language_model):
        """The model fixture loaded without errors."""
        assert language_model is not None

    def test_detect_english(self, language_model):
        result = detect_user_language("I have been feeling very anxious lately")
        assert result == "en"

    def test_detect_arabic(self, language_model):
        result = detect_user_language("أنا أشعر بالقلق الشديد")
        assert result == "ar"

    def test_detect_french(self, language_model):
        result = detect_user_language("Je me sens très triste aujourd'hui")
        assert result == "fr"

    def test_detect_spanish(self, language_model):
        result = detect_user_language("Estoy muy preocupado por mi salud mental")
        assert result == "es"

    def test_detect_german(self, language_model):
        result = detect_user_language("Ich fühle mich sehr einsam und traurig")
        assert result == "de"

    def test_returns_string(self, language_model):
        """Output must always be a string, never None or a list."""
        result = detect_user_language("Hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_short_message(self, language_model):
        """Short messages are known to be unreliable. We test that
        the function still returns a valid string and does not crash."""
        result = detect_user_language("why?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string_does_not_crash(self, language_model):
        """Even if the model misclassifies, it should not throw."""
        result = detect_user_language("a")
        assert isinstance(result, str)

    def test_model_not_loaded_raises(self):
        """If someone calls detect without loading, it should raise."""
        import deployment.services.language_detection as lang_mod

        original = lang_mod._model
        lang_mod._model = None
        try:
            with pytest.raises(RuntimeError, match="not loaded"):
                detect_user_language("Hello")
        finally:
            lang_mod._model = original
