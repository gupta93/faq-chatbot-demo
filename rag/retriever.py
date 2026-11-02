from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunker import Chunk, chunk_all_documents
from rag.loader import load_documents


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class DocumentRetriever:
    def __init__(
        self,
        data_dir: str = "data",
        top_k: int = 4,
        min_score: float = 0.08,
    ) -> None:
        self.data_dir = data_dir
        self.top_k = top_k
        self.min_score = min_score
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._build_index()

    def _build_index(self) -> None:
        from pathlib import Path

        documents = load_documents(Path(self.data_dir))
        self._chunks = chunk_all_documents(documents)
        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            return
        corpus = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not self._chunks or self._vectorizer is None or self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).ravel()
        ranked = sorted(
            (
                RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
                for i in range(len(self._chunks))
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        top = ranked[: self.top_k]
        return [r for r in top if r.score >= self.min_score]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
