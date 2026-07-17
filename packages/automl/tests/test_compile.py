"""Tests for benchcraft_automl.compile.

Covers the package's one signature capability for this scaffold-depth
pass: fusing a fitted sklearn Pipeline into a single ONNX graph, and
verifying that graph produces the same predictions as the original
pipeline when run through onnxruntime.

Requires the `onnx` extra (skl2onnx, onnx, onnxruntime) to be installed;
tests are skipped (not failed) if it isn't, since that extra is optional
by design.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import benchcraft_automl
from benchcraft_automl import CompileOptions, ONNXExtraNotInstalledError, compile

onnxruntime = pytest.importorskip(
    "onnxruntime", reason="onnxruntime not installed; skipping onnx-dependent tests"
)
onnx = pytest.importorskip("onnx")


def _fit_scaler_logreg_pipeline(random_state: int = 0) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=random_state,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test.astype(np.float32)


def _run_onnx(onnx_model, X: np.ndarray) -> np.ndarray:
    session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X.astype(np.float32)})
    return outputs


def test_public_api_surface():
    assert benchcraft_automl.compile is compile
    assert hasattr(benchcraft_automl, "CompileOptions")
    assert hasattr(benchcraft_automl, "ONNXExtraNotInstalledError")


def test_compile_rejects_non_pipeline():
    with pytest.raises(TypeError):
        compile(LogisticRegression(), np.zeros((1, 3)))


def test_compile_rejects_unfitted_pipeline():
    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("clf", LogisticRegression())])
    with pytest.raises(NotFittedError):
        compile(pipeline, np.zeros((1, 3)))


def test_compile_produces_valid_onnx_model():
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    onnx_model = compile(pipeline, X_test)
    assert isinstance(onnx_model, onnx.ModelProto)
    onnx.checker.check_model(onnx_model)


def test_compiled_onnx_predictions_match_sklearn_predict():
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    onnx_model = compile(pipeline, X_test)

    sklearn_preds = pipeline.predict(X_test)
    onnx_outputs = _run_onnx(onnx_model, X_test)
    onnx_labels = onnx_outputs[0]

    assert np.array_equal(onnx_labels.astype(sklearn_preds.dtype), sklearn_preds)


def test_compiled_onnx_predictions_match_sklearn_predict_proba():
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    onnx_model = compile(pipeline, X_test, options=CompileOptions(zipmap=False))

    sklearn_proba = pipeline.predict_proba(X_test)
    onnx_outputs = _run_onnx(onnx_model, X_test)
    onnx_proba = np.asarray(onnx_outputs[1])

    assert np.allclose(onnx_proba, sklearn_proba, atol=1e-4)


def test_compile_accepts_pandas_dataframe_input():
    pd = pytest.importorskip("pandas")
    pipeline, X_train, X_test = _fit_scaler_logreg_pipeline()
    df_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])

    onnx_model = compile(pipeline, df_test)
    sklearn_preds = pipeline.predict(X_test)
    onnx_labels = _run_onnx(onnx_model, X_test)[0]

    assert np.array_equal(onnx_labels.astype(sklearn_preds.dtype), sklearn_preds)


def test_compile_raises_typeerror_for_non_numeric_dataframe_column():
    pd = pytest.importorskip("pandas")
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()

    # A DataFrame with a non-numeric ("string") column cannot be coerced to
    # a float32 array. The documented contract (compile()'s docstring) says
    # this raises TypeError, not a raw ValueError from pandas/numpy.
    bad_df = pd.DataFrame({"f": ["a", "b", "c"]})

    with pytest.raises(TypeError) as exc_info:
        compile(pipeline, bad_df)

    message = str(exc_info.value)
    assert "float32" in message
    assert "f" in message


def test_compile_on_breast_cancer_dataset_end_to_end():
    data = load_breast_cancer()
    X_train, X_test, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )
    pipeline = Pipeline(
        steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    pipeline.fit(X_train, y_train)

    onnx_model = compile(pipeline, X_test)
    sklearn_preds = pipeline.predict(X_test)
    onnx_labels = _run_onnx(onnx_model, X_test.astype(np.float32))[0]

    assert np.array_equal(onnx_labels.astype(sklearn_preds.dtype), sklearn_preds)
