import pytest

from rag.answer import REFUSAL, answer_question
from rag.llm import LLMConfig
from rag.retriever import DocumentRetriever

DATA_DIR = "data"
MOCK_CONFIG = LLMConfig(
    api_key=None,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    mock=True,
)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def retriever():
    return DocumentRetriever(data_dir=DATA_DIR, min_score=0.08)


def test_retrieval_finds_relevant_plan_pricing(retriever):
    hits = retriever.retrieve("How much does the Enthusiast subscription cost?")
    assert hits, "expected at least one relevant chunk for pricing question"
    combined = " ".join(h.chunk.text.lower() for h in hits)
    assert "32" in combined or "enthusiast" in combined


def test_refusal_when_no_relevant_docs(retriever):
    hits = retriever.retrieve("What is the orbital period of Neptune?")
    assert hits == []
    result = answer_question("What is the orbital period of Neptune?", retriever, MOCK_CONFIG)
    assert result.refused is True
    assert result.text == REFUSAL
    assert result.citations == []


def test_shipping_question_grounded_in_policy(retriever):
    result = answer_question("Do you ship to Canada?", retriever, MOCK_CONFIG)
    assert not result.refused
    assert any("shipping" in c.doc_id for c in result.citations)


def test_citations_on_successful_answer(retriever):
    result = answer_question("How do I pause my subscription?", retriever, MOCK_CONFIG)
    assert not result.refused
    assert len(result.citations) >= 1
    doc_ids = {c.doc_id for c in result.citations}
    assert any("beanvoyage" in d or "shipping" in d for d in doc_ids)
    assert all(c.section for c in result.citations)
    assert "[Mock mode]" in result.text
