"""Document-grounded FAQ retrieval and answering."""

from rag.answer import answer_question
from rag.retriever import DocumentRetriever

__all__ = ["DocumentRetriever", "answer_question"]
