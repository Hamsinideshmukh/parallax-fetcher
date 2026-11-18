"""Semantic search over the chunks: embed each chunk once, then rank by cosine similarity."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # small (about 130 MB), runs fine on a laptop CPU


class Embedder(Protocol):
    """Anything that turns text into vectors. Tests use a fake one, so they need no model."""

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...
    def embed_query(self, text: str) -> Vectors: ...


class FastEmbedder:
    """A local embedding model through fastembed: ONNX on the CPU, no GPU or API key needed."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # imported here so `fetcher fetch` starts fast

        self.model = TextEmbedding(model_name)  # downloads the model once, then uses the cache

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return np.array(list(self.model.embed(list(texts))), dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        # bge models embed queries with a short instruction prefix, which query_embed adds
        return np.array(next(iter(self.model.query_embed(text))), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    url: str
    title: str | None
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, from -1 to 1, higher is closer


def read_chunks(path: Path) -> list[Chunk]:
    """Read the chunks.jsonl file that scripts/to_chunks.py writes."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def normalize(vectors: Vectors) -> Vectors:
    """Scale every vector to length 1, so a dot product is the cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


class SearchIndex:
    def __init__(self, chunks: list[Chunk], vectors: Vectors) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = normalize(vectors)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "SearchIndex":
        return cls(chunks, embedder.embed_documents([c.text for c in chunks]))

    def search(self, query_vector: Vectors, k: int = 5, one_per_page: bool = True) -> list[Hit]:
        """The k closest chunks. With one_per_page, a long page can't fill every slot."""
        scores = self.vectors @ normalize(query_vector)
        hits: list[Hit] = []
        seen_urls: set[str] = set()
        for i in np.argsort(-scores):  # best score first
            chunk = self.chunks[int(i)]
            if one_per_page and chunk.url in seen_urls:
                continue
            seen_urls.add(chunk.url)
            hits.append(Hit(chunk, float(scores[i])))
            if len(hits) == k:
                break
        return hits

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")

    @classmethod
    def load(cls, folder: Path) -> "SearchIndex":
        vectors: Vectors = np.load(folder / "vectors.npy")
        return cls(read_chunks(folder / "chunks.jsonl"), vectors)
