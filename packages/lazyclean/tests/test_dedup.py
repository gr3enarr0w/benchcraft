"""Tests for cosine-similarity near-duplicate detection over embeddings.

Threshold choice (0.9): with the hashing bag-of-words preprocessor, two
sentences that differ only in punctuation/case tokenize to the *identical*
bag of words, so their embeddings are effectively identical (cosine
similarity ~1.0). A sentence about a completely different topic shares
almost no vocabulary, so its embedding direction is essentially unrelated
(cosine similarity well below 0.9 in practice, typically near/at 0 for a
fixed random linear projection of two disjoint feature-hash supports).
0.9 sits comfortably between those two regimes for this scaffold's fixture,
without being so close to 1.0 that ordinary token-order/whitespace noise
would produce false negatives on true near-duplicates.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchcraft_lazyclean import detect_near_duplicate_text
from benchcraft_lazyclean.dedup import (
    DedupReport,
    DuplicatePair,
    cosine_similarity_matrix,
    find_near_duplicates,
)
from benchcraft_lazyclean.embeddings import build_synthetic_embedding_model

NEAR_DUP_THRESHOLD = 0.9

ROWS = [
    "The quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog!!",  # near-duplicate of row 0
    "Quantum entanglement enables non-local correlations between particles",  # distinct
]


@pytest.fixture()
def model(tmp_path):
    """Build a small synthetic ONNX embedding model, cached under ``tmp_path``."""
    return build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=128, embedding_dim=32)


def test_cosine_similarity_matrix_shape_and_diagonal():
    """A cosine-similarity matrix is (n, n), has a unit diagonal, and rates
    identical vectors as similarity 1.0 and orthogonal vectors as 0.0."""
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    sims = cosine_similarity_matrix(embeddings)
    assert sims.shape == (3, 3)
    np.testing.assert_allclose(np.diag(sims), 1.0, atol=1e-6)
    assert sims[0, 2] == pytest.approx(1.0, abs=1e-6)  # identical vectors
    assert sims[0, 1] == pytest.approx(0.0, abs=1e-6)  # orthogonal vectors


def test_cosine_similarity_matrix_zero_vectors_are_self_similar_and_mutually_duplicate():
    """Regression test: an all-zero embedding row (e.g. empty/whitespace/
    punctuation-only text through hashing_bag_of_words_vectorizer) used to
    compute cosine similarity 0.0 against itself and against an identical
    zero row, because the old implementation clamped zero norms to 1.0
    before normalizing, leaving the zero vector as the zero vector. Two
    zero vectors are identical rows and must read as duplicates (1.0); a
    zero vector against a genuinely non-zero vector must NOT look similar.
    """
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],  # zero row A ("")
            [0.0, 0.0, 0.0],  # zero row B ("   ") -- identical zero vector
            [1.0, 0.0, 0.0],  # genuinely non-zero row
        ],
        dtype=np.float32,
    )
    sims = cosine_similarity_matrix(embeddings)

    # Self-similarity of a zero vector is 1.0, not 0.0.
    assert sims[0, 0] == pytest.approx(1.0)
    assert sims[1, 1] == pytest.approx(1.0)

    # Two identical zero vectors are duplicates of each other.
    assert sims[0, 1] == pytest.approx(1.0)
    assert sims[1, 0] == pytest.approx(1.0)

    # A zero vector against a genuinely non-zero vector is not similar --
    # the fix must not make everything look similar to a zero vector.
    assert sims[0, 2] == pytest.approx(0.0)
    assert sims[1, 2] == pytest.approx(0.0)


def test_find_near_duplicates_flags_identical_zero_vector_rows():
    """find_near_duplicates() flags two identical all-zero embedding rows as
    a duplicate pair with similarity 1.0, without sweeping in a distinct
    non-zero row."""
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    report = find_near_duplicates(embeddings, threshold=NEAR_DUP_THRESHOLD)

    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)  # identical zero-vector rows are duplicates
    assert 2 not in flagged  # the non-zero row is not swept in as a false positive

    assert any(
        {pair.index_a, pair.index_b} == {0, 1} and pair.similarity == pytest.approx(1.0)
        for pair in report.pairs
    )


def test_cosine_similarity_matrix_rejects_non_2d_input():
    """A 1-D array (not an (n_rows, dim) matrix) raises ValueError."""
    with pytest.raises(ValueError):
        cosine_similarity_matrix(np.zeros(5, dtype=np.float32))


def test_find_near_duplicates_rejects_bad_threshold():
    """Thresholds outside the valid (0.0, 1.0] range raise ValueError."""
    embeddings = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError):
        find_near_duplicates(embeddings, threshold=0.0)
    with pytest.raises(ValueError):
        find_near_duplicates(embeddings, threshold=1.5)


def test_find_near_duplicates_flags_the_near_duplicate_pair(model):
    """End-to-end via the synthetic model: the near-duplicate ROWS[0]/ROWS[1]
    pair is flagged above threshold, and the unrelated ROWS[2] is not."""
    embeddings = model.embed(ROWS)
    report = find_near_duplicates(embeddings, threshold=NEAR_DUP_THRESHOLD)

    assert isinstance(report, DedupReport)
    assert report.num_rows == 3
    assert report.threshold == NEAR_DUP_THRESHOLD

    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)  # the near-duplicate pair is flagged
    assert 2 not in flagged  # the distinct row is not flagged

    assert any(
        {pair.index_a, pair.index_b} == {0, 1} and pair.similarity >= NEAR_DUP_THRESHOLD
        for pair in report.pairs
    )


def test_dedup_report_pairs_sorted_by_descending_similarity(model):
    """DedupReport.pairs is sorted by descending similarity, most-similar
    pair first."""
    embeddings = model.embed(ROWS + ["the quick brown fox jumps over the lazy dog"])
    report = find_near_duplicates(embeddings, threshold=0.1)
    similarities = [pair.similarity for pair in report.pairs]
    assert similarities == sorted(similarities, reverse=True)


def test_detect_near_duplicate_text_end_to_end(model):
    """detect_near_duplicate_text() embeds plain string rows and returns
    both the embeddings and a report flagging the expected near-duplicate
    pair."""
    embeddings, report = detect_near_duplicate_text(ROWS, model, threshold=NEAR_DUP_THRESHOLD)
    assert embeddings.shape == (3, model.embedding_dim)
    assert {0, 1}.issubset(report.flagged_indices())
    assert 2 not in report.flagged_indices()


def test_detect_near_duplicate_text_accepts_arrow_backed_pandas_series(model):
    """detect_near_duplicate_text() accepts a Tier-1 Arrow-backed pandas
    Series (ArrowDtype) directly, without needing to convert to a plain
    list first."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(ROWS).convert_dtypes(dtype_backend="pyarrow")
    embeddings, report = detect_near_duplicate_text(series, model, threshold=NEAR_DUP_THRESHOLD)
    assert embeddings.shape == (3, model.embedding_dim)
    assert {0, 1}.issubset(report.flagged_indices())
    assert 2 not in report.flagged_indices()


def test_detect_near_duplicate_text_flags_empty_and_whitespace_only_text_as_duplicates(model):
    """End-to-end regression test for the concrete failure scenario: two
    genuinely empty-text rows ("" and "   ") tokenize to zero tokens via
    hashing_bag_of_words_vectorizer, so both embed to the identical all-zero
    vector through the real EmbeddingModel/ONNX path (not hand-constructed
    arrays). They must be flagged as near-duplicates, and the distinct
    non-empty row must not be swept in.
    """
    rows = ["", "   ", "The quick brown fox jumps over the lazy dog"]
    embeddings, report = detect_near_duplicate_text(rows, model, threshold=NEAR_DUP_THRESHOLD)

    # Sanity check that these two rows really do embed to the same all-zero
    # vector via the real preprocessor + ONNX model, i.e. this test actually
    # exercises the bug scenario rather than some other coincidental match.
    np.testing.assert_array_equal(embeddings[0], np.zeros(model.embedding_dim, dtype=np.float32))
    np.testing.assert_array_equal(embeddings[1], np.zeros(model.embedding_dim, dtype=np.float32))

    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)
    assert 2 not in flagged


def test_duplicate_pair_is_frozen_dataclass():
    """DuplicatePair is immutable: assigning to a field after construction
    raises."""
    pair = DuplicatePair(index_a=0, index_b=1, similarity=0.99)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.similarity = 0.5  # type: ignore[misc]
