"""Tests for query routing logic."""
import pytest
from lexar.retrieval.query_router import QueryRouter


@pytest.fixture
def router():
    return QueryRouter()


def test_ipc_keyword_routes_to_ipc(router):
    result = router.route("What is IPC section 302?")
    assert result["ipc"] is True


def test_judgment_keyword_routes_to_judgment(router):
    result = router.route("What did the Supreme Court hold in this case?")
    assert result["judgment"] is True


def test_user_docs_requires_flag(router):
    result = router.route("explain clause 5 of uploaded document", has_user_docs=False)
    assert result["user"] is False

    result = router.route("explain clause 5 of uploaded document", has_user_docs=True)
    assert result["user"] is True


def test_ambiguous_routes_to_both(router):
    result = router.route("explain murder in India")
    assert result["ipc"] is True or result["judgment"] is True


def test_default_fallback_routes_to_ipc_and_judgment(router):
    result = router.route("hello")
    assert result["ipc"] is True
    assert result["judgment"] is True


def test_section_number_triggers_ipc(router):
    result = router.route("tell me about 420")
    assert result["ipc"] is True


def test_route_returns_dict_with_required_keys(router):
    result = router.route("test query")
    assert "ipc" in result
    assert "judgment" in result
    assert "user" in result
