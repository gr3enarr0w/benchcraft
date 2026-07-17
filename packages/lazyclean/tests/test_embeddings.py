"""Tests for the PyTorch-free ONNX Runtime embedding path.

Fully hermetic: uses build_synthetic_embedding_model() (a hand-built ONNX
graph via the `onnx` package's graph-builder API), never touches the
network, and never bundles a real multi-hundred-MB model file.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from benchcraft_lazyclean import MODEL_ALLOWLIST, RECOMMENDED_MODEL_NAME
from benchcraft_lazyclean.embeddings import (
    EmbeddingModel,
    build_synthetic_embedding_model,
    build_synthetic_embedding_onnx,
    hashing_bag_of_words_vectorizer,
)
from lazycore.licensing import ModelTier


def test_no_pytorch_or_transformers_imported():
    """The hard constraint: this package must never pull in torch/transformers."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_build_synthetic_embedding_onnx_writes_a_valid_model(tmp_path):
    onnx_path = tmp_path / "tiny.onnx"
    result = build_synthetic_embedding_onnx(onnx_path, vocab_dim=64, embedding_dim=16, seed=1)

    assert result == onnx_path
    assert onnx_path.exists()
    # A hand-built graph fixture, not a bundled real model -- expect it to
    # be tiny (a few KB), not multi-hundred-MB.
    assert onnx_path.stat().st_size < 100_000


def test_build_synthetic_embedding_model_returns_working_embedding_model(tmp_path):
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)

    assert isinstance(model, EmbeddingModel)
    assert model.embedding_dim == 16

    embeddings = model.embed(["hello world", "goodbye world"])
    assert embeddings.shape == (2, 16)
    assert embeddings.dtype == np.float32


def test_embed_output_is_l2_normalized(tmp_path):
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)
    embeddings = model.embed(["some sample text", "another distinct sample"])
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_embed_is_deterministic_across_calls(tmp_path):
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)
    first = model.embed(["repeat this exact sentence"])
    second = model.embed(["repeat this exact sentence"])
    np.testing.assert_array_equal(first, second)


def test_embed_empty_input_returns_empty_array(tmp_path):
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)
    embeddings = model.embed([])
    assert embeddings.shape == (0, 16)


def test_hashing_bag_of_words_vectorizer_is_deterministic_and_shaped():
    vectorize = hashing_bag_of_words_vectorizer(vocab_dim=32)
    vec_a = vectorize("The Quick Brown Fox")
    vec_b = vectorize("the quick brown fox")  # case-insensitive tokenization
    assert vec_a.shape == (32,)
    np.testing.assert_array_equal(vec_a, vec_b)


def test_hashing_bag_of_words_vectorizer_empty_string_is_zero_vector():
    vectorize = hashing_bag_of_words_vectorizer(vocab_dim=16)
    vec = vectorize("   ")
    np.testing.assert_array_equal(vec, np.zeros(16, dtype=np.float32))


def test_recommended_model_registered_as_tier_1():
    entry = MODEL_ALLOWLIST.get(RECOMMENDED_MODEL_NAME)
    assert entry is not None
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "Apache-2.0"


def test_from_onnx_file_infers_input_output_names(tmp_path):
    onnx_path = build_synthetic_embedding_onnx(
        tmp_path / "infer_names.onnx", vocab_dim=32, embedding_dim=8
    )
    preprocessor = hashing_bag_of_words_vectorizer(vocab_dim=32)
    model = EmbeddingModel.from_onnx_file(
        onnx_path, preprocessor=preprocessor, embedding_dim=8
    )
    assert model.input_name == "input"
    assert model.output_name == "embedding"
