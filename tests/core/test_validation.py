"""Input and option validation, and the edge cases that used to crash."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from chapkit_deep_auto_regressive.core import DeepAutoRegressiveModel, ModelOptions
from chapkit_deep_auto_regressive.core.trainer import LossFn, Trainer, TrainState
from chapkit_deep_auto_regressive.core.transforms import get_series, period_start_date
from chapkit_deep_auto_regressive.train import _read_config


def _frame(periods: list[str], n_locations: int = 2, seed: int = 0) -> pd.DataFrame:
    """A small synthetic panel over the given periods."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        [
            {
                "time_period": period,
                "location": f"loc{location}",
                "disease_cases": float(rng.poisson(20)),
                "rainfall": float(rng.normal(100, 20)),
                "mean_temperature": 20.0,
                "population": 1000.0 * (location + 1),
            }
            for location in range(n_locations)
            for period in periods
        ]
    )


def _months(n: int) -> list[str]:
    return [f"{2000 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]


def _model(**options: Any) -> DeepAutoRegressiveModel:
    """A small, fast model configuration."""
    defaults: dict[str, Any] = {"context_length": 6, "prediction_length": 2, "n_ensemble": 1, "n_iter": 5}
    return DeepAutoRegressiveModel(ModelOptions(**{**defaults, **options}))


@pytest.mark.parametrize(
    ("periods", "bad"),
    [
        (["2020-01", "2020-02", "2020-02"], "'2020-02' is followed by '2020-02'"),
        (["2020-01", "2020-02", "2020-04"], "'2020-02' is followed by '2020-04'"),
        (["2020W01", "2020W02", "2020W04"], "'2020W02' is followed by '2020W04'"),
    ],
)
def test_get_series_rejects_duplicate_or_missing_periods(periods: list[str], bad: str) -> None:
    with pytest.raises(ValueError, match="consecutive") as error:
        get_series(_frame(periods))
    assert bad in str(error.value)


def test_get_series_accepts_consecutive_weeks_across_a_year_boundary() -> None:
    # 2020 has 53 ISO weeks.
    x, _ = get_series(_frame(["2020W52", "2020W53", "2021W01", "2021-W02"]))
    assert x.shape[1] == 4


def test_get_series_accepts_a_skipped_iso_week_53() -> None:
    # chapkit's test data, and some other sources, number every year 1-52.
    x, _ = get_series(_frame(["2020-W51", "2020-W52", "2021-W01", "2021-W02"]))
    assert x.shape[1] == 4


def test_get_series_rejects_infinite_features_and_invalid_case_counts() -> None:
    data = _frame(_months(4))
    data.loc[0, "rainfall"] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        get_series(data)
    data = _frame(_months(4))
    data.loc[0, "disease_cases"] = -1.0
    with pytest.raises(ValueError, match="finite and not negative"):
        get_series(data)
    # Missing case counts stay allowed.
    data = _frame(_months(4))
    data.loc[0, "disease_cases"] = np.nan
    get_series(data)


@pytest.mark.parametrize("period", ["2020-01-01/2020-01-31", "2020-01-07/2020-01-01"])
def test_a_date_range_must_be_one_week(period: str) -> None:
    with pytest.raises(ValueError, match="span 7 days"):
        period_start_date(period)


def test_a_date_range_must_hold_real_dates() -> None:
    with pytest.raises(ValueError):
        period_start_date("2020-01-01/1999-99-99")


@pytest.mark.parametrize(
    "options",
    [
        {"eval_every": 0},
        {"patience": 0},
        {"n_ensemble": 0},
        {"dropout_rate": 1.0},
        {"input_dropout_rate": -0.1},
        {"rnn_features": 0},
        {"rnn_layers": 0},
        {"warmup_epochs": -1},
    ],
)
def test_invalid_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelOptions(**options)


def test_a_one_period_recursive_forecast_trains_and_predicts() -> None:
    data = _frame(_months(30))
    predictor = _model(prediction_length=1, recursive_decode=True).train(data)
    history = data[data["time_period"] < "2002-06"]
    future = data[data["time_period"] == "2002-06"].drop(columns="disease_cases")
    samples = predictor.predict(history, future, num_samples=5)
    assert len(samples) == len(future)


def test_training_that_ends_with_non_finite_weights_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Trainer.train

    def nan_like(leaf: jax.Array) -> jax.Array:
        return jnp.full_like(leaf, jnp.nan)

    def diverging(self: Trainer, loader: Any, loss_fn: LossFn) -> TrainState:
        state = original(self, loader, loss_fn)
        return state.replace(params=jax.tree_util.tree_map(nan_like, state.params))

    monkeypatch.setattr(Trainer, "train", diverging)
    with pytest.raises(RuntimeError, match="training diverged"):
        _model().train(_frame(_months(30)))


def test_a_missing_config_file_is_an_error(tmp_path: Path) -> None:
    assert _read_config(None).user_option_values is None
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _read_config(str(tmp_path / "typo.yml"))
