import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    text: str


def _section_title(block: str) -> str:
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or "Introduction"
        if stripped:
            return stripped[:80] + ("…" if len(stripped) > 80 else "")
    return "Introduction"


def chunk_document(doc_id: str, text: str, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split document into overlapping chunks, preserving section hints from markdown."""
    text = text.strip()
    if not text:
        return []

    blocks = re.split(r"\n(?=#{1,3}\s)", text)
    chunks: list[Chunk] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        section = _section_title(block)
        if len(block) <= max_chars:
            chunks.append(Chunk(doc_id=doc_id, section=section, text=block))
            continue
        start = 0
        while start < len(block):
            end = min(start + max_chars, len(block))
            piece = block[start:end].strip()
            if piece:
                chunks.append(Chunk(doc_id=doc_id, section=section, text=piece))
            if end >= len(block):
                break
            start = max(0, end - overlap)
    return chunks


def chunk_all_documents(documents: list[tuple[str, str]], **kwargs) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc_id, text in documents:
        all_chunks.extend(chunk_document(doc_id, text, **kwargs))
    return all_chunks
