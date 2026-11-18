from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from fetcher.search import Chunk, SearchIndex, Vectors

WORDS = ["csv", "regex", "date"]


class BagOfWordsEmbedder:
    """A fake embedder: one dimension per word in WORDS. Enough to test ranking, no model needed."""

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        return np.array([text.split().count(w) for w in WORDS], dtype=np.float32)


CHUNKS = [
    Chunk("a#0", "https://a.com", "A", "csv csv rows"),
    Chunk("a#1", "https://a.com", "A", "csv and a date"),
    Chunk("b#0", "https://b.com", "B", "regex patterns"),
    Chunk("c#0", "https://c.com", "C", "date and time"),
]


@pytest.fixture
def index() -> SearchIndex:
    return SearchIndex.build(CHUNKS, BagOfWordsEmbedder())


def test_closest_chunk_ranks_first(index: SearchIndex) -> None:
    hits = index.search(BagOfWordsEmbedder().embed_query("regex"), k=2)
    assert hits[0].chunk.id == "b#0"
    assert hits[0].score == pytest.approx(1.0)


def test_one_chunk_per_page_by_default(index: SearchIndex) -> None:
    query = BagOfWordsEmbedder().embed_query("csv")
    assert [h.chunk.url for h in index.search(query, k=3)].count("https://a.com") == 1
    assert [h.chunk.id for h in index.search(query, k=2, one_per_page=False)] == ["a#0", "a#1"]


def test_save_and_load_round_trip(index: SearchIndex, tmp_path: Path) -> None:
    index.save(tmp_path)
    loaded = SearchIndex.load(tmp_path)
    assert loaded.chunks == CHUNKS
    assert np.allclose(loaded.vectors, index.vectors)


def test_mismatched_chunks_and_vectors_are_rejected() -> None:
    with pytest.raises(ValueError):
        SearchIndex(CHUNKS, np.zeros((2, 3), dtype=np.float32))
