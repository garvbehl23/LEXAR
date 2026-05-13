"""Unit tests for LexarPipeline with mocked components."""
import pytest
from unittest.mock import MagicMock, patch


def _make_chunk(section: str, text: str):
    return {
        "chunk_id": f"IPC_{section}",
        "text": text,
        "score": 0.9,
        "metadata": {"statute": "IPC", "section": section, "jurisdiction": "India"},
    }


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        _make_chunk("302", "Whoever commits murder shall be punished with death."),
        _make_chunk("304", "Whoever commits culpable homicide shall be punished."),
    ]
    return retriever


@pytest.fixture
def mock_reranker():
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, chunks, k: sorted(
        [dict(c, rerank_score=0.8) for c in chunks], key=lambda x: -x["rerank_score"]
    )[:k]
    return reranker


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate_with_evidence.return_value = {
        "answer": "Murder is punishable with death or life imprisonment.",
        "error": None,
        "provenance": {},
        "evidence_token_count": 50,
        "query_token_count": 10,
        "attention_mask_shape": (60, 60),
    }
    return gen


def test_pipeline_no_evidence_returns_no_evidence_status():
    from lexar.lexar_pipeline import LexarPipeline

    pipeline = LexarPipeline()
    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = []
    pipeline.retriever = mock_ret

    result = pipeline.answer("What is murder?")
    assert result["status"] == "no_evidence"
    assert result["evidence_count"] == 0


def test_pipeline_success_path(mock_retriever, mock_reranker, mock_generator):
    from lexar.lexar_pipeline import LexarPipeline

    pipeline = LexarPipeline()
    pipeline.retriever = mock_retriever
    pipeline.reranker = mock_reranker
    pipeline.generator = mock_generator

    result = pipeline.answer("What is the punishment for murder?")
    assert result["status"] == "success"
    assert len(result["answer"]) > 0
    assert result["evidence_count"] > 0
    assert result["confidence"] > 0.0


def test_pipeline_result_has_evidence_ids(mock_retriever, mock_reranker, mock_generator):
    from lexar.lexar_pipeline import LexarPipeline

    pipeline = LexarPipeline()
    pipeline.retriever = mock_retriever
    pipeline.reranker = mock_reranker
    pipeline.generator = mock_generator

    result = pipeline.answer("What is murder?")
    assert "evidence_ids" in result
    assert len(result["evidence_ids"]) > 0
