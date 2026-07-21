"""dscraft.automl: DSCraft's clean-room tabular AutoML module.

This scaffold-depth pass implements the architecture doc's (Part 3,
"Module 1: AutoML") `.compile()` path that fuses a fitted
`sklearn.pipeline.Pipeline` into a single ONNX graph via `skl2onnx`,
replacing brittle pickle-based serialization (see `compile.py` and the
package README for details and clean-room provenance), plus a
multi-backend model-selection surface: pluggable gradient-boosted-tree
model backends (`models.py`: XGBoost/LightGBM/CatBoost, per the
multi-backend design principle -- all three equally supported, none a
"default").

The streaming/incremental `partial_fit` fading-factor evaluator and the
PSI drift-detection feature from the same architecture doc section are
explicitly out of scope for this pass -- future work, not implemented
here.

Public API surface (this is the one canonical entrypoint per capability --
no parallel export path exists elsewhere in this package):
    >>> from dscraft.automl import compile, CompileOptions
    >>> from dscraft.automl import build_model, SUPPORTED_CLASSIFIERS, SUPPORTED_REGRESSORS
"""

from dscraft.automl.compile import (
    CompileOptions,
    ONNXExtraNotInstalledError,
    compile,  # noqa: A004 -- intentional public API name, see architecture doc
)
from dscraft.automl.models import SUPPORTED_CLASSIFIERS, SUPPORTED_REGRESSORS, build_model

__all__ = [
    "compile",
    "CompileOptions",
    "ONNXExtraNotInstalledError",
    "SUPPORTED_CLASSIFIERS",
    "SUPPORTED_REGRESSORS",
    "build_model",
]

__version__ = "0.1.0"
