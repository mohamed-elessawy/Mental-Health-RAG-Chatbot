"""
Tests for the RAG pipeline.

Both Qdrant and LLM calls are mocked. We test the pipeline
logic: query rewriting, document retrieval flow, response
generation, and the full rag_answer orchestration.
"""

from unittest.mock import MagicMock, patch

import pytest

from deployment.schemas.prompts import (
    get_generation_system_prompt,
    get_query_rewrite_prompt,
)
from deployment.services.rag_service import (
    generate_response,
    rag_answer,
    retrieve_documents,
    rewrite_query,
)


class TestRewriteQuery:
    def _mock_rewrite_response(self, context: str, query: str):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = f"CONTEXT: {context}\nQUERY: {query}"
        return mock_response

    def test_extracts_context_and_query(self):
        with patch("deployment.services.rag_service.litellm.completion") as mock:
            mock.return_value = self._mock_rewrite_response(
                "female, in a relationship",
                "coping with anxiety in relationships",
            )
            query, context = rewrite_query("My boyfriend makes me anxious")
            assert query == "coping with anxiety in relationships"
            assert context == "female, in a relationship"

    def test_no_context_found(self):
        with patch("deployment.services.rag_service.litellm.completion") as mock:
            mock.return_value = self._mock_rewrite_response(
                "none", "dealing with general anxiety"
            )
            query, context = rewrite_query("I feel anxious all the time")
            assert context == "none"
            assert len(query) > 0

    def test_llm_failure_propagates(self):
        with patch("deployment.services.rag_service.litellm.completion") as mock:
            mock.side_effect = RuntimeError("API timeout")
            with pytest.raises(RuntimeError):
                rewrite_query("test message")


class TestRetrieveDocuments:
    def test_returns_formatted_results(self):
        mock_point = MagicMock()
        mock_point.payload = {
            "question": "How to deal with anxiety?",
            "responses": ["Try deep breathing", "Consider therapy"],
            "topics": ["anxiety"],
        }

        mock_results = MagicMock()
        mock_results.points = [mock_point]

        with (
            patch("deployment.services.rag_service.embedder") as mock_embedder,
            patch("deployment.services.rag_service.qdrant") as mock_qdrant,
        ):
            mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384
            mock_qdrant.query_points.return_value = mock_results

            results = retrieve_documents("anxiety coping strategies")
            assert len(results) == 1
            assert results[0]["question"] == "How to deal with anxiety?"
            assert len(results[0]["responses"]) == 2

    def test_empty_results(self):
        mock_results = MagicMock()
        mock_results.points = []

        with (
            patch("deployment.services.rag_service.embedder") as mock_embedder,
            patch("deployment.services.rag_service.qdrant") as mock_qdrant,
        ):
            mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384
            mock_qdrant.query_points.return_value = mock_results

            results = retrieve_documents("something very obscure")
            assert results == []

    def test_not_initialized_raises(self):
        with (
            patch("deployment.services.rag_service.embedder", None),
            patch("deployment.services.rag_service.qdrant", None),
        ):
            with pytest.raises(RuntimeError, match="not initialized"):
                retrieve_documents("test")


class TestGenerateResponse:
    def test_generates_with_context(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I understand your anxiety."

        retrieved = [
            {
                "question": "How to cope?",
                "responses": ["Try breathing exercises"],
                "topics": ["anxiety"],
            }
        ]

        with patch("deployment.services.rag_service.litellm.completion") as mock:
            mock.return_value = mock_response
            result = generate_response(
                user_message="I feel anxious",
                retrieved=retrieved,
                emotion="fear",
                personal_context="none",
            )
            assert result == "I understand your anxiety."

    def test_includes_history_in_messages(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        with patch("deployment.services.rag_service.litellm.completion") as mock:
            mock.return_value = mock_response
            generate_response(
                user_message="Help me",
                retrieved=[],
                history=history,
            )
            # Check that history messages were included in the call
            call_messages = mock.call_args[1]["messages"]
            # system + 2 history + 1 user = 4
            assert len(call_messages) == 4


class TestRagAnswer:
    def test_full_pipeline(self):
        """Test the complete orchestration: rewrite -> retrieve -> generate."""
        mock_rewrite = MagicMock()
        mock_rewrite.choices = [MagicMock()]
        mock_rewrite.choices[0].message.content = "CONTEXT: none\nQUERY: anxiety help"

        mock_generate = MagicMock()
        mock_generate.choices = [MagicMock()]
        mock_generate.choices[0].message.content = "Here is some help."

        mock_point = MagicMock()
        mock_point.payload = {
            "question": "How to deal with anxiety?",
            "responses": ["Try deep breathing"],
            "topics": ["anxiety"],
        }
        mock_results = MagicMock()
        mock_results.points = [mock_point]

        with (
            patch("deployment.services.rag_service.litellm.completion") as mock_llm,
            patch("deployment.services.rag_service.embedder") as mock_embedder,
            patch("deployment.services.rag_service.qdrant") as mock_qdrant,
        ):
            # First call is rewrite, second is generate
            mock_llm.side_effect = [mock_rewrite, mock_generate]
            mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384
            mock_qdrant.query_points.return_value = mock_results

            result = rag_answer("I feel anxious", emotion="fear")

            assert result["answer"] == "Here is some help."
            assert result["search_query"] == "anxiety help"
            assert result["personal_context"] == "none"
            assert len(result["sources"]) == 1

    def test_returns_dict_with_required_keys(self):
        mock_rewrite = MagicMock()
        mock_rewrite.choices = [MagicMock()]
        mock_rewrite.choices[0].message.content = "CONTEXT: none\nQUERY: test"

        mock_generate = MagicMock()
        mock_generate.choices = [MagicMock()]
        mock_generate.choices[0].message.content = "Response"

        mock_results = MagicMock()
        mock_results.points = []

        with (
            patch("deployment.services.rag_service.litellm.completion") as mock_llm,
            patch("deployment.services.rag_service.embedder") as mock_embedder,
            patch("deployment.services.rag_service.qdrant") as mock_qdrant,
        ):
            mock_llm.side_effect = [mock_rewrite, mock_generate]
            mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384
            mock_qdrant.query_points.return_value = mock_results

            result = rag_answer("test", emotion="neutral")
            assert "answer" in result
            assert "search_query" in result
            assert "personal_context" in result
            assert "sources" in result


class TestPromptConstruction:
    def test_rewrite_prompt_contains_message(self):
        prompt = get_query_rewrite_prompt("I feel anxious about work")
        assert "I feel anxious about work" in prompt
        assert "CONTEXT:" in prompt
        assert "QUERY:" in prompt

    def test_generation_prompt_contains_emotion(self):
        prompt = get_generation_system_prompt("sadness", "female", "some context")
        assert "sadness" in prompt
        assert "female" in prompt
        assert "some context" in prompt
