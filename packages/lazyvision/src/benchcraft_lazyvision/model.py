"""A small CNN image classifier (architecture doc Part 3, "Module 5: LazyVision").

Scope for this scaffold-depth pass is deliberately narrow: a tiny,
LeNet-style CNN (a few conv layers, one linear head), used purely to prove
the `torch.export` -> ONNX export path in :mod:`benchcraft_lazyvision.export`
works end-to-end and correctly. Training a real model on real data is
explicitly **not** required at this scope -- the point is validating the
export mechanism, not model accuracy -- so :func:`build_model` returns an
initialized-but-untrained model. Vision Transformers, real-time object
detectors (YOLO/D-FINE/RT-DETR), and acoustic/spectrogram models from the
same architecture-doc section are out of scope for this pass; see the
package README's "Deferred" section.

Per CLAUDE.md's MPS-primary constraint: nothing here hardcodes
``device="cpu"`` in a way that would prevent later running on MPS --
:func:`build_model` accepts an arbitrary ``torch.device``/device string and
the model itself is plain `nn.Module` layers with no CUDA-only or
CPU-only assumptions. This module's own tests/examples default to CPU
purely because that is the fastest, most portable choice for automated,
hermetic verification -- not because MPS is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "TinyCNN",
    "ModelConfig",
    "build_model",
    "synthetic_classification_batch",
]


@dataclass(frozen=True)
class ModelConfig:
    """Shape configuration shared between :class:`TinyCNN` and its callers.

    Kept as a small dataclass (rather than scattering magic numbers across
    ``model.py``, ``export.py``, and the tests/example) so the CNN's input
    contract has exactly one definition.
    """

    in_channels: int = 3
    image_size: int = 32
    num_classes: int = 10


class TinyCNN(nn.Module):
    """A minimal LeNet-style CNN: two conv+pool blocks, one linear head.

    Deliberately small (a "few conv layers", per the task's scope) --
    this is a scaffold proving the export path, not a competitive image
    classifier. Input is expected to be ``(N, in_channels, image_size,
    image_size)`` float32, e.g. the dense tensor produced by
    :class:`benchcraft_lazyvision.pipeline.SimpleImagePipeline`.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Build the conv/pool/linear layers for the given ``config``.

        Args:
            config: Shape configuration (input channels, image size, number
                of output classes). Defaults to ``ModelConfig()``.

        Raises:
            ValueError: ``config.image_size`` is too small for two 2x2
                max-pools to leave at least a 1x1 feature map (i.e. less
                than 4).
        """
        super().__init__()
        self.config = config or ModelConfig()

        self.conv1 = nn.Conv2d(self.config.in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Two 2x2 max-pools halve the spatial size twice.
        reduced = self.config.image_size // 4
        if reduced < 1:
            raise ValueError(
                "ModelConfig.image_size must be >= 4 so that two 2x2 "
                f"max-pools leave at least a 1x1 feature map; got "
                f"{self.config.image_size}."
            )
        self.fc = nn.Linear(16 * reduced * reduced, self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two conv+relu+pool blocks and the linear head.

        Args:
            x: Input batch of shape ``(N, in_channels, image_size,
                image_size)``, float32.

        Returns:
            Unnormalized class logits of shape ``(N, num_classes)``.
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def build_model(
    config: ModelConfig | None = None,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> TinyCNN:
    """Construct a :class:`TinyCNN` with deterministic (seeded) init weights.

    Not trained -- per this pass's scope, an initialized-but-untrained model
    is sufficient to validate the export path (see module docstring).

    Args:
        config: Optional :class:`ModelConfig`. Defaults to ``ModelConfig()``
            (3x32x32 input, 10 classes).
        seed: Seed for ``torch.manual_seed`` so weight init is reproducible
            across test runs.
        device: Where to place the model. Defaults to ``"cpu"`` for this
            module's tests/examples (fastest, most portable for automated
            verification). Per CLAUDE.md, MPS (``torch.device("mps")``) is
            the platform's intended primary backend -- callers are free to
            pass that in instead; nothing here assumes CPU.

    Returns:
        A :class:`TinyCNN` in ``eval()`` mode on ``device``.
    """
    generator_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = TinyCNN(config)
    finally:
        torch.set_rng_state(generator_state)
    return model.to(device).eval()


def synthetic_classification_batch(
    config: ModelConfig | None = None,
    *,
    batch_size: int = 4,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small, fully-synthetic (images, labels) batch.

    Stands in for a real image-classification dataset so that tests and the
    example script stay hermetic and fast (no network access, no dataset
    download), per the task's "prefer whichever keeps tests hermetic/fast"
    guidance. Images are random float32 tensors in ``[0, 1)``; labels are
    random integers in ``[0, num_classes)``. This is not meant to represent
    a learnable task -- it is purely input/output-shape-compatible stand-in
    data for exercising the pipeline and export path.

    Returns:
        ``(images, labels)`` where ``images`` has shape
        ``(batch_size, in_channels, image_size, image_size)`` and ``labels``
        has shape ``(batch_size,)``.
    """
    cfg = config or ModelConfig()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.rand(
        (batch_size, cfg.in_channels, cfg.image_size, cfg.image_size),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    labels = torch.randint(
        low=0,
        high=cfg.num_classes,
        size=(batch_size,),
        generator=generator,
    ).to(device)
    return images, labels
