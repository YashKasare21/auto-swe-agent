"""FAISS-based vector store for code chunk retrieval.

Uses inner-product (cosine similarity on L2-normalised vectors).
Falls back to brute-force numpy search when FAISS is unavailable.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from indexing.parser import CodeChunk

DEFAULT_INDEX_PATH = str(Path(__file__).parent / "code_index.faiss")


class CodeVectorStore:
    """FAISS index + metadata for code chunk similarity search."""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix(".pkl")
        self.dimension = 384  # all-MiniLM-L6-v2 output dim
        self._index = None
        self.metadata: List[CodeChunk] = []
        self._use_fallback = False

    def build(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None:
        """Build index from chunks and their embeddings."""
        self.metadata = chunks
        try:
            import faiss
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            self._index = faiss.IndexFlatIP(self.dimension)
            self._index.add(embeddings)
            self._use_fallback = False
        except ImportError:
            warnings.warn("faiss not available — using brute-force numpy search")
            self._use_fallback = True
            self._fallback_embeddings = embeddings.copy()
        self.save()

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[CodeChunk, float]]:
        """Return top-k (chunk, cosine_similarity) matches."""
        if not self.metadata:
            return []

        if not self._use_fallback and self._index is None:
            self.load()

        query = query_embedding.astype(np.float32).reshape(1, -1)

        if self._use_fallback or self._index is None:
            return self._fallback_search(query, k)

        import faiss
        faiss.normalize_L2(query)
        distances, indices = self._index.search(query, k)
        results: List[Tuple[CodeChunk, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        return results

    def _fallback_search(
        self, query: np.ndarray, k: int
    ) -> List[Tuple[CodeChunk, float]]:
        """Brute-force cosine similarity when FAISS is unavailable."""
        if not hasattr(self, "_fallback_embeddings"):
            return []
        query_norm = query / (np.linalg.norm(query) + 1e-12)
        emb_norm = self._fallback_embeddings / (
            np.linalg.norm(self._fallback_embeddings, axis=1, keepdims=True) + 1e-12
        )
        scores = emb_norm @ query_norm.T
        scores = scores.flatten()
        top_k = min(k, len(scores))
        indices = np.argsort(-scores)[:top_k]
        results: List[Tuple[CodeChunk, float]] = []
        for idx in indices:
            results.append((self.metadata[idx], float(scores[idx])))
        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self._use_fallback:
            try:
                import faiss
                faiss.write_index(self._index, str(self.index_path))
            except Exception:
                pass
        # Always save metadata and fallback embeddings
        payload = {
            "metadata": self.metadata,
            "fallback_embeddings": getattr(self, "_fallback_embeddings", None),
        }
        self.metadata_path.write_bytes(pickle.dumps(payload))

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True on success."""
        if not self.index_path.exists() and not self.metadata_path.exists():
            return False

        # Load metadata
        if self.metadata_path.exists():
            try:
                payload = pickle.loads(self.metadata_path.read_bytes())
                self.metadata = payload.get("metadata", [])
                fb_emb = payload.get("fallback_embeddings")
                if fb_emb is not None:
                    self._fallback_embeddings = fb_emb
                    self._use_fallback = True
            except Exception:
                return False

        # Load FAISS index
        if self.index_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self.index_path))
                self._use_fallback = False
                return True
            except Exception:
                pass

        return bool(self.metadata)

    def index_exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()
