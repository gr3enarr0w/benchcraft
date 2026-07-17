"""Hermetic tests for benchcraft_lazyforecast.forecast.

No network access required. Builds a small synthetic multi-series dataset
(sine-wave-plus-trend, fixed seed, >=2 distinct series IDs) directly in this
file, fits AutoARIMA over it, and asserts basic sanity on the resulting
forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchcraft_lazyforecast import ForecastConfig, forecast, prepare_frame, validate_input
from benchcraft_lazyforecast.forecast import SUPPORTED_MODELS


def _make_synthetic_panel(n_points: int = 120, seed: int = 42) -> pd.DataFrame:
    """Two seasonal-plus-trend series with a fixed seed, for hermetic testing.

    Series "series_a" and "series_b" each have a distinct trend slope and
    seasonal amplitude/phase, plus small Gaussian noise, over `n_points`
    daily observations with a 7-day season.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)

    frames = []
    series_params = {
        "series_a": {"trend": 0.05, "amplitude": 5.0, "phase": 0.0, "level": 50.0},
        "series_b": {"trend": -0.03, "amplitude": 8.0, "phase": 1.5, "level": 100.0},
    }
    for unique_id, params in series_params.items():
        seasonal = params["amplitude"] * np.sin(2 * np.pi * t / 7 + params["phase"])
        trend = params["trend"] * t
        noise = rng.normal(scale=0.5, size=n_points)
        y = params["level"] + trend + seasonal + noise
        frames.append(pd.DataFrame({"unique_id": unique_id, "ds": dates, "y": y}))

    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    return _make_synthetic_panel()


@pytest.fixture(scope="module")
def arrow_backed_panel(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """The same panel, converted to Tier-1 Arrow-backed pandas (ArrowDtype)."""
    return synthetic_panel.convert_dtypes(dtype_backend="pyarrow")


def test_supported_models_are_classical_only() -> None:
    assert set(SUPPORTED_MODELS) == {"AutoARIMA", "AutoETS"}


def test_forecast_config_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        ForecastConfig(models=("LightGBM",))


def test_forecast_config_rejects_empty_models() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ForecastConfig(models=())


def test_validate_input_reports_arrow_backed_columns(arrow_backed_panel: pd.DataFrame) -> None:
    report = validate_input(arrow_backed_panel)
    assert report.input_kind == "pandas"
    assert report.n_series == 2
    assert report.is_fully_arrow_backed is True
    assert report.warnings == []


def test_validate_input_warns_on_non_arrow_backed(synthetic_panel: pd.DataFrame) -> None:
    report = validate_input(synthetic_panel)
    assert report.is_fully_arrow_backed is False
    assert any("Arrow-backed" in w for w in report.warnings)


def test_validate_input_missing_column_raises(synthetic_panel: pd.DataFrame) -> None:
    broken = synthetic_panel.drop(columns=["y"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_input(broken)


def test_validate_input_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        validate_input([1, 2, 3])


def test_prepare_frame_produces_expected_schema(arrow_backed_panel: pd.DataFrame) -> None:
    prepared = prepare_frame(arrow_backed_panel)
    assert list(prepared.columns) == ["unique_id", "ds", "y"]
    assert prepared["ds"].dtype.kind == "M"  # datetime64
    assert prepared["y"].dtype == np.float64
    assert prepared["unique_id"].nunique() == 2
    assert np.isfinite(prepared["y"].to_numpy()).all()


def test_prepare_frame_rejects_nan_values(synthetic_panel: pd.DataFrame) -> None:
    broken = synthetic_panel.copy()
    broken.loc[0, "y"] = float("nan")
    with pytest.raises(ValueError, match="NaN/inf"):
        prepare_frame(broken)


def test_forecast_autoarima_shape_and_finiteness(arrow_backed_panel: pd.DataFrame) -> None:
    config = ForecastConfig(horizon=7, freq="D", season_length=7, models=("AutoARIMA",))
    result = forecast(arrow_backed_panel, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA"}
    # 2 series * 7-step horizon = 14 forecast rows.
    assert len(result) == 2 * config.horizon
    for unique_id in ("series_a", "series_b"):
        subset = result[result["unique_id"] == unique_id]
        assert len(subset) == config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()


def test_forecast_multiple_models(arrow_backed_panel: pd.DataFrame) -> None:
    config = ForecastConfig(horizon=5, freq="D", season_length=7, models=("AutoARIMA", "AutoETS"))
    result = forecast(arrow_backed_panel, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA", "AutoETS"}
    assert len(result) == 2 * config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
    assert np.isfinite(result["AutoETS"].to_numpy()).all()


def test_forecast_accepts_plain_pandas_without_arrow_dtype(synthetic_panel: pd.DataFrame) -> None:
    """Non-Arrow-backed pandas input is still usable -- Tier-1 is a
    convention we validate/report on, not a hard gate (see README)."""
    config = ForecastConfig(horizon=5, models=("AutoARIMA",))
    result = forecast(synthetic_panel, config)
    assert len(result) == 2 * config.horizon


def test_forecast_accepts_custom_column_names(synthetic_panel: pd.DataFrame) -> None:
    renamed = synthetic_panel.rename(columns={"unique_id": "series_id", "ds": "timestamp", "y": "value"})
    config = ForecastConfig(
        id_col="series_id", time_col="timestamp", value_col="value", horizon=5, models=("AutoARIMA",)
    )
    result = forecast(renamed, config)
    assert len(result) == 2 * config.horizon


def test_forecast_accepts_polars_input(synthetic_panel: pd.DataFrame) -> None:
    pl = pytest.importorskip("polars")
    polars_panel = pl.from_pandas(synthetic_panel)

    report = validate_input(polars_panel)
    assert report.input_kind == "polars"
    assert report.n_series == 2

    config = ForecastConfig(horizon=5, models=("AutoARIMA",))
    result = forecast(polars_panel, config)
    assert len(result) == 2 * config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
