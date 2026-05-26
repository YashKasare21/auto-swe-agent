"""Code embedding using sentence-transformers.

Embeds code chunks into 384-dim vectors using all-MiniLM-L6-v2.
Falls back to a simple TF-IDF-like bag-of-words embedding if sentence-transformers
is unavailable (e.g. on first run before download completes).
"""

from __future__ import annotations

import re
import warnings
from typing import List, Optional

import numpy as np

from indexing.parser import CodeChunk


class CodeEmbedder:
    """Wraps a sentence-transformer model for code embedding."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_vocab: dict[str, int] = {}
        self._use_fallback = False

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except (ImportError, OSError, Exception) as exc:
            warnings.warn(
                f"sentence-transformers not available ({exc}). "
                f"Using fallback bag-of-words embeddings."
            )
            self._use_fallback = True
            self._build_fallback_vocab()

    def _build_fallback_vocab(self):
        """Build a simple vocabulary for fallback embeddings."""
        common_tokens = (
            "def class import from return if else for while try except "
            "with as async await yield lambda self super init str int "
            "float bool list dict set tuple none true false raise pass "
            "break continue and or not in is assert global nonlocal "
            "del print len range open read write get set add append "
            "pop remove clear copy sort reverse find index split join "
            "replace strip format startswith endswith encode decode "
        )
        self._fallback_vocab = {t: i for i, t in enumerate(common_tokens.split())}
        # Extend with common programming terms
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz_"):
            self._fallback_vocab.setdefault(c, len(self._fallback_vocab))

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Simple bag-of-words fallback embedding (384-dim)."""
        embeddings = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.lower())
            for token in tokens:
                idx = self._fallback_vocab.get(token, hash(token) % 384)
                if idx < 384:
                    embeddings[i, idx] += 1.0
            # Normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings into vectors (384-dim float32)."""
        self._load_model()
        if self._use_fallback:
            return self._fallback_encode(texts)
        return self._model.encode(texts, show_progress_bar=False)

    def embed_chunk(self, chunk: CodeChunk) -> np.ndarray:
        """Create a rich text representation of a code chunk and embed it."""
        text = (
            f"{chunk.name}\n"
            f"{chunk.signature}\n"
            f"{chunk.docstring}\n"
            f"{chunk.body_preview}"
        )
        return self.embed([text])[0]
