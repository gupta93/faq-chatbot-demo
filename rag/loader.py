from pathlib import Path

from pypdf import PdfReader


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Load all supported documents from data_dir recursively.

    Returns list of (source_label, text) where source_label is relative path.
    """
    if not data_dir.is_dir():
        return []

    docs: list[tuple[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel = str(path.relative_to(data_dir))
        try:
            if suffix in {".md", ".txt"}:
                docs.append((rel, load_text_file(path)))
            elif suffix == ".pdf":
                docs.append((rel, load_pdf_file(path)))
        except OSError:
            continue
    return docs
