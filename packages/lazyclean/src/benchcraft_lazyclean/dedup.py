"""Cosine-similarity near-duplicate detection over embeddings.

This is a minimal stand-in for the Density-Based Semantic Deduplication
(D4) capability described in the architecture doc's Part 3 ("Module 2:
LazyClean"). The real D4 design uses spherical mini-batch k-means to bucket
embeddings into clusters, then an IVF-HNSW approximate-nearest-neighbor
index within/across clusters, specifically to *avoid* O(n^2) pairwise
cosine-similarity cost at scale.

**This module deliberately implements only the naive O(n^2) brute-force
version.** For a small batch this is a perfectly correct and simple
stand-in for the ANN index; it does not scale past a few thousand rows
before the pairwise similarity matrix becomes the bottleneck the
architecture doc calls out IVF-HNSW as solving. Replacing
:func:`cosine_similarity_matrix` (and the O(n^2) loop in
:func:`find_near_duplicates`) with a real IVF-HNSW index is tracked as
follow-up work, not implemented in this scaffold-depth pass -- see the
package README for the explicit scope boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DuplicatePair",
    "DedupReport",
    "cosine_similarity_matrix",
    "find_near_duplicates",
]


@dataclass(frozen=True)
class DuplicatePair:
    """One flagged near-duplicate row pair."""

    index_a: int
    index_b: int
    similarity: float


@dataclass
class DedupReport:
    """Result of a near-duplicate scan over a batch of embedded rows."""

    pairs: list[DuplicatePair]
    threshold: float
    num_rows: int

    def flagged_indices(self) -> set[int]:
        """The set of row indices that appear in at least one flagged pair."""
        flagged: set[int] = set()
        for pair in self.pairs:
            flagged.add(pair.index_a)
            flagged.add(pair.index_b)
        return flagged


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the full pairwise cosine-similarity matrix for ``embeddings``.

    **Naive O(n^2) implementation.** This materializes an ``(n, n)`` dense
    similarity matrix, which is the exact cost profile the architecture
    doc's IVF-HNSW approximate-nearest-neighbor index (Part 3, "Module 2:
    LazyClean") exists to avoid at scale. Acceptable and simple for the
    small batches this scaffold-depth pass targets; not a substitute for
    that index once real dataset sizes are in play.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D (n_rows, dim) array, got shape {embeddings.shape!r}.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    is_zero_vector = (norms == 0.0).reshape(-1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = embeddings / safe_norms
    similarities = normalized @ normalized.T

    # A genuinely all-zero embedding row (e.g. hashing_bag_of_words_vectorizer
    # on empty/whitespace-only/punctuation-only text) has no defined
    # direction, so the plain normalized-dot-product above always reads 0.0
    # for it -- including against itself. For *pairwise duplicate detection*
    # that is the wrong practical answer: two identical zero vectors are
    # identical rows and must be flagged as duplicates (similarity 1.0),
    # while a zero vector against a genuinely non-zero vector should stay at
    # 0.0 (not similar). Patch in that special case explicitly rather than
    # relying on the undefined cosine-similarity behavior.
    if is_zero_vector.any():
        zero_pair_mask = np.outer(is_zero_vector, is_zero_vector)
        similarities = np.where(zero_pair_mask, 1.0, similarities)
    return similarities


def find_near_duplicates(embeddings: np.ndarray, *, threshold: float = 0.92) -> DedupReport:
    """Flag row-index pairs whose cosine similarity is at or above ``threshold``.

    Brute-force O(n^2): computes the full pairwise similarity matrix via
    :func:`cosine_similarity_matrix` and scans its upper triangle. See the
    module docstring for why this is an intentional, documented scope
    boundary rather than the production-scale IVF-HNSW path.

    Args:
        embeddings: ``(n_rows, dim)`` array, typically from
            :meth:`benchcraft_lazyclean.embeddings.EmbeddingModel.embed`.
        threshold: cosine-similarity cutoff in ``(0.0, 1.0]`` above which a
            pair is flagged as a near-duplicate.

    Returns:
        A :class:`DedupReport` with pairs sorted by descending similarity.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold!r}.")

    num_rows = embeddings.shape[0]
    similarities = cosine_similarity_matrix(embeddings)

    pairs: list[DuplicatePair] = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            similarity = float(similarities[i, j])
            if similarity >= threshold:
                pairs.append(DuplicatePair(index_a=i, index_b=j, similarity=similarity))

    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return DedupReport(pairs=pairs, threshold=threshold, num_rows=num_rows)
