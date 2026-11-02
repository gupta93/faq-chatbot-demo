from dataclasses import dataclass

from rag.llm import LLMConfig, generate_answer, load_llm_config
from rag.retriever import DocumentRetriever, RetrievedChunk

REFUSAL = "I don't know."


@dataclass(frozen=True)
class Citation:
    doc_id: str
    section: str
    score: float


@dataclass(frozen=True)
class AnswerResult:
    text: str
    citations: list[Citation]
    refused: bool


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, r in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Document: {r.chunk.doc_id} | Section: {r.chunk.section}\n{r.chunk.text}"
        )
    return "\n\n".join(parts)


def citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(doc_id=r.chunk.doc_id, section=r.chunk.section, score=r.score) for r in chunks
    ]


def answer_question(
    question: str,
    retriever: DocumentRetriever,
    config: LLMConfig | None = None,
) -> AnswerResult:
    config = config or load_llm_config()
    retrieved = retriever.retrieve(question)

    if not retrieved:
        return AnswerResult(text=REFUSAL, citations=[], refused=True)

    context = format_context(retrieved)
    raw = generate_answer(question, context, config=config)
    refused = REFUSAL.lower() in raw.lower() and len(raw) < 40

    if refused:
        return AnswerResult(text=REFUSAL, citations=[], refused=True)

    return AnswerResult(
        text=raw,
        citations=citations_from_chunks(retrieved),
        refused=False,
    )
