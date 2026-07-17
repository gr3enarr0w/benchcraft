"""Tests for the minimal GCN forward pass (`benchcraft_lazygraph.gcn.GCN`).

Builds a small synthetic graph via `PyGSparseAdapter`, feeds it random
node features through `GCN`, and verifies the output has the expected
shape and is entirely finite (no NaN/Inf from a broken forward pass).
"""

from __future__ import annotations

import torch

from benchcraft_lazygraph import GCN, PyGSparseAdapter, resolve_device


NUM_NODES = 10
IN_CHANNELS = 5
HIDDEN_CHANNELS = 8
OUT_CHANNELS = 3


def _make_adapter() -> PyGSparseAdapter:
    # Small ring graph, made bidirectional (undirected) so every node has
    # at least one neighbor for message passing.
    src = list(range(NUM_NODES)) + list(range(1, NUM_NODES)) + [0]
    dst = list(range(1, NUM_NODES)) + [0] + list(range(NUM_NODES - 1))
    # (src -> next node) and (dst -> previous node), i.e. both directions.
    forward_src = list(range(NUM_NODES))
    forward_dst = [(i + 1) % NUM_NODES for i in range(NUM_NODES)]
    all_src = forward_src + forward_dst
    all_dst = forward_dst + forward_src
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    return PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def test_gcn_forward_pass_output_shape_and_finiteness():
    torch.manual_seed(0)
    device = resolve_device()

    adapter = _make_adapter()
    x = torch.randn(NUM_NODES, IN_CHANNELS, device=device)

    model = GCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x, adapter)

    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()


def test_gcn_accepts_csr_backed_adapter_too():
    """The adapter's native format shouldn't matter to the GCN -- it
    converts to COO internally via `.to_coo()`."""
    torch.manual_seed(1)
    adapter = _make_adapter().to_csr()

    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = GCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    model.eval()

    with torch.no_grad():
        out = model(x, adapter)

    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()


def test_resolve_device_falls_back_cleanly_for_bogus_preference():
    device = resolve_device(preferred="not-a-real-device")
    assert device.type in ("cpu", "mps", "cuda")


def test_resolve_device_default_never_raises():
    device = resolve_device()
    assert isinstance(device, torch.device)
