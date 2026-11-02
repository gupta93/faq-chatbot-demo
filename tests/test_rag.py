import logging
from pathlib import Path

import pytest

from rag.answer import REFUSAL, answer_question
from rag.llm import LLMConfig
from rag.loader import load_documents
from rag.retriever import DocumentRetriever

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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
    return DocumentRetriever(data_dir=str(DATA_DIR), min_score=0.08)


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


def test_load_documents_skips_corrupt_pdf(tmp_path, caplog):
    (tmp_path / "notes.txt").write_text("Valid note.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\n% corrupted content")

    with caplog.at_level(logging.WARNING):
        docs = load_documents(tmp_path)

    assert docs == [("notes.txt", "Valid note.")]
    assert any("broken.pdf" in record.message for record in caplog.records)


def test_load_documents_includes_sample_pdf():
    docs = load_documents(DATA_DIR)
    pdf_docs = [text for name, text in docs if name.endswith(".pdf")]
    assert pdf_docs, "expected at least one PDF in data/"
    combined = " ".join(pdf_docs).lower()
    assert "gift" in combined or "25" in combined
