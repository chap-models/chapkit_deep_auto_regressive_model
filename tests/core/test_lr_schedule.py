"""The optional warmup + cosine-decay learning-rate schedule."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from chapkit_deep_auto_regressive.core import DeepAutoRegressiveModel


def _frame(n_periods: int, n_locations: int = 3, seed: int = 0) -> pd.DataFrame:
    """A small synthetic panel in the shape the model expects."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for location in range(n_locations):
        for period in range(n_periods):
            month = period % 12
            rate = float(20 + 10 * np.sin(2 * np.pi * month / 12))
            rows.append(
                {
                    "time_period": f"{2000 + period // 12}-{month + 1:02d}",
                    "location": f"loc{location}",
                    "disease_cases": float(rng.poisson(rate)),
                    "rainfall": float(rng.normal(100, 20)),
                    "mean_temperature": float(20 + 5 * np.sin(2 * np.pi * month / 12)),
                    "population": 1000.0 * (location + 1),
                }
            )
    return pd.DataFrame(rows)


def _model(**options: Any) -> DeepAutoRegressiveModel:
    """A small, fast model configuration."""
    model = DeepAutoRegressiveModel()
    model.options.context_length = 6
    model.options.prediction_length = 2
    model.options.n_ensemble = 1
    model.options.n_iter = 10
    model.options.eval_every = 2
    model.options.patience = 3
    model.options.validation_periods = 4
    for key, value in options.items():
        setattr(model.options, key, value)
    return model


def test_warmup_cosine_schedule_shape() -> None:
    model = _model(lr_schedule="warmup_cosine", n_iter=100, learning_rate=1e-5)
    steps_per_epoch = 10
    schedule = model.learning_rate_for(steps_per_epoch)
    assert callable(schedule)
    assert np.isclose(schedule(0), 1e-5)  # starts at the base rate
    assert np.isclose(schedule(5 * steps_per_epoch), 1e-3)  # peak after the 5 default warmup epochs
    assert np.isclose(schedule(100 * steps_per_epoch), 1e-5)  # decayed back by the end of n_iter
    assert np.isclose(schedule(1000 * steps_per_epoch), 1e-5)  # and stays there


def test_constant_schedule_is_the_default_and_unknown_schedule_fails() -> None:
    assert _model(learning_rate=1e-4).learning_rate_for(10) == 1e-4
    with pytest.raises(ValueError, match="lr_schedule"):
        _model(lr_schedule="cosine").learning_rate_for(10)


@pytest.mark.parametrize("early_stopping", [False, True])
def test_train_and_predict_with_warmup_cosine_schedule(early_stopping: bool) -> None:
    data = _frame(40)
    model = _model(lr_schedule="warmup_cosine", warmup_epochs=2, early_stopping=early_stopping)
    predictor = model.train(data)
    history = data[data["time_period"] < "2003-03"]
    future = data[data["time_period"].isin(["2003-03", "2003-04"])].drop(columns="disease_cases")
    samples = predictor.predict(history, future, num_samples=5)
    sample_cols = [c for c in samples.columns if c.startswith("sample_")]
    assert len(sample_cols) == 5
    assert len(samples) == len(future)
    assert np.isfinite(samples[sample_cols].to_numpy()).all()
    # The schedule is applied per trainer and never replaces the configured rate.
    assert model.options.learning_rate == 1e-3
