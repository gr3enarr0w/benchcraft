"""Tests for `PyGSparseAdapter` (Tier-2 sparse graph tensor adapter).

Constructs a small synthetic graph (8 nodes, a handful of directed edges)
directly, and verifies that `.to_coo()` / `.to_csr()` / `.to_csc()` all
round-trip to a structurally equivalent dense adjacency matrix -- i.e. the
conversion is real (produces correct data), not just a metadata relabel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchcraft_lazygraph import PyGSparseAdapter
from lazycore.data import SparseFormat, SparseGraphTensorAdapter


NUM_NODES = 8
# A small ring plus a couple of chords, directed edges (src, dst).
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
    (0, 4), (2, 6),
]


def _make_adapter() -> PyGSparseAdapter:
    src = [e[0] for e in EDGES]
    dst = [e[1] for e in EDGES]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def _expected_dense() -> np.ndarray:
    dense = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for s, d in EDGES:
        dense[s, d] = 1.0
    return dense


def test_is_concrete_subclass_of_lazycore_adapter():
    adapter = _make_adapter()
    assert isinstance(adapter, SparseGraphTensorAdapter)


def test_native_format_and_shape_on_construction():
    adapter = _make_adapter()
    assert adapter.native_format == SparseFormat.COO
    assert adapter.shape == (NUM_NODES, NUM_NODES)


def test_coo_round_trip_matches_expected_dense():
    adapter = _make_adapter()
    coo_adapter = adapter.to_coo()
    assert coo_adapter.native_format == SparseFormat.COO
    np.testing.assert_array_equal(coo_adapter.to_dense_numpy(), _expected_dense())


def test_csr_conversion_matches_expected_dense():
    adapter = _make_adapter()
    csr_adapter = adapter.to_csr()
    assert csr_adapter.native_format == SparseFormat.CSR
    np.testing.assert_array_equal(csr_adapter.to_dense_numpy(), _expected_dense())
    # Confirm it's a real scipy CSR matrix, not a relabel.
    import scipy.sparse as sp

    assert isinstance(csr_adapter.scipy_matrix, sp.csr_matrix)


def test_csc_conversion_matches_expected_dense():
    adapter = _make_adapter()
    csc_adapter = adapter.to_csc()
    assert csc_adapter.native_format == SparseFormat.CSC
    np.testing.assert_array_equal(csc_adapter.to_dense_numpy(), _expected_dense())
    import scipy.sparse as sp

    assert isinstance(csc_adapter.scipy_matrix, sp.csc_matrix)


def test_csr_to_coo_round_trip_preserves_structure():
    adapter = _make_adapter()
    csr_adapter = adapter.to_csr()
    back_to_coo = csr_adapter.to_coo()
    assert back_to_coo.native_format == SparseFormat.COO
    np.testing.assert_array_equal(back_to_coo.to_dense_numpy(), _expected_dense())


def test_csc_to_coo_round_trip_preserves_structure():
    adapter = _make_adapter()
    csc_adapter = adapter.to_csc()
    back_to_coo = csc_adapter.to_coo()
    assert back_to_coo.native_format == SparseFormat.COO
    np.testing.assert_array_equal(back_to_coo.to_dense_numpy(), _expected_dense())


def test_all_three_formats_agree_on_dense_reconstruction():
    adapter = _make_adapter()
    dense_coo = adapter.to_coo().to_dense_numpy()
    dense_csr = adapter.to_csr().to_dense_numpy()
    dense_csc = adapter.to_csc().to_dense_numpy()
    np.testing.assert_array_equal(dense_coo, dense_csr)
    np.testing.assert_array_equal(dense_coo, dense_csc)


def test_invalid_edge_index_shape_rejected():
    bad_edge_index = torch.tensor([0, 1, 2], dtype=torch.long)
    with pytest.raises(ValueError):
        PyGSparseAdapter.from_edge_index(bad_edge_index, num_nodes=NUM_NODES)


def test_edge_index_accessor_requires_coo_format():
    adapter = _make_adapter()
    csr_adapter = adapter.to_csr()
    with pytest.raises(RuntimeError):
        _ = csr_adapter.edge_index


def test_scipy_matrix_accessor_requires_csr_or_csc_format():
    adapter = _make_adapter()
    with pytest.raises(RuntimeError):
        _ = adapter.scipy_matrix
