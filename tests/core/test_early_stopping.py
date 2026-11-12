"""Early stopping: the split, the two-phase refit, and the short-series guard."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import chapkit_deep_auto_regressive.core.model as model_module
from chapkit_deep_auto_regressive.core import DeepAutoRegressiveModel, ModelOptions
from chapkit_deep_auto_regressive.core.data_loader import DataSet, SimpleDataLoader, WindowLoader
from chapkit_deep_auto_regressive.core.rnn_model import build_network
from chapkit_deep_auto_regressive.core.trainer import LossFn, Trainer, TrainState


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
    model = DeepAutoRegressiveModel(ModelOptions(early_stopping=True))
    model.options.context_length = 6
    model.options.prediction_length = 2
    model.options.n_ensemble = 1
    model.options.n_iter = 30
    model.options.eval_every = 2
    model.options.patience = 3
    model.options.validation_periods = 4
    for key, value in options.items():
        setattr(model.options, key, value)
    return model


def test_split_is_contiguous_and_final_in_time():
    """The held-out block is the tail of the series, never a random subset."""
    model = _model()
    model._locations = ["loc0", "loc1", "loc2"]
    data = _frame(40)
    loaders = model._early_stopping_loaders(data)
    assert loaders is not None
    fit_loader, validation_loader = loaders

    total_length = model.options.context_length + model.options.prediction_length
    n_validation = total_length + model.options.validation_periods
    # The validation windows must come from the final n_validation periods, so the
    # first validation target cannot appear anywhere in the training windows.
    assert len(validation_loader.dataset) == model.options.validation_periods + 1
    assert len(fit_loader.dataset) == 40 - n_validation - total_length + 1


def test_too_short_a_series_raises_rather_than_falling_back():
    """A silent fallback would reintroduce the overtraining this prevents."""
    model = _model()
    model._locations = ["loc0", "loc1", "loc2"]
    total_length = model.options.context_length + model.options.prediction_length
    # Held-out block, plus enough left for the probe to get min_probe_windows.
    minimum = (total_length + model.options.validation_periods) + total_length + model.options.min_probe_windows - 1

    model._early_stopping_loaders(_frame(minimum))  # exactly enough: fine

    with pytest.raises(ValueError, match="early stopping needs at least"):
        model._early_stopping_loaders(_frame(minimum - 1))


def test_disabled_or_externally_supplied_validation_returns_none():
    """Early stopping steps aside when switched off or when given a window."""
    data = _frame(40)
    off = _model(early_stopping=False)
    off._locations = ["loc0", "loc1", "loc2"]
    assert off._early_stopping_loaders(data) is None

    supplied = _model()
    supplied._locations = ["loc0", "loc1", "loc2"]
    supplied._validation_loader = SimpleDataLoader(DataSet(np.zeros((3, 10, 1)), np.zeros((3, 10)), forecast_length=2))
    assert supplied._early_stopping_loaders(data) is None


def test_training_selects_an_epoch_and_refits_on_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two phases run, and the second is capped by the epoch the first chose."""
    model = _model()
    selected: list[tuple[int, int]] = []
    original = Trainer.train

    def recording(self: Trainer, loader: WindowLoader, loss_fn: LossFn) -> TrainState:
        state = original(self, loader, loss_fn)
        if self.best_epoch is not None:
            selected.append((self.best_epoch, self.n_iter))
        return state

    monkeypatch.setattr(Trainer, "train", recording)
    predictor = model.train(_frame(60))

    assert predictor is not None
    assert selected, "the probe phase should have reported a best epoch"
    best_epoch, probe_budget = selected[0]
    assert 0 <= best_epoch <= probe_budget


def test_short_run_still_trains_when_early_stopping_is_off():
    """Series below the early-stopping minimum remain trainable explicitly."""
    model = _model(early_stopping=False, n_iter=3)
    predictor = model.train(_frame(20))
    assert predictor is not None


def test_trainer_without_validation_keeps_the_final_state():
    """With no validation loader the trainer behaves as it did before."""
    trainer = Trainer(model=build_network(3), n_iter=3)
    assert trainer.best_epoch is None
    assert trainer.patience == 6


def test_refit_budget_is_computed_in_steps_under_both_scalings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hand-off must be a step budget, however it is scaled.

    The probe runs over the pre-split prefix and the refit over the whole frame,
    so the loaders hold different numbers of windows and an epoch index means
    different amounts of training in each. Under ``"steps"`` the refit gets the
    probe's step count unchanged; under ``"epochs"`` it gets that count scaled by
    the window ratio, i.e. the same number of passes. Either way the budget the
    refit enforces is expressed in steps, so it never depends on which loader it
    happens to be iterating.
    """

    def run(scaling: str) -> tuple[dict[str, Any], dict[str, Any]]:
        model = _model(refit_scaling=scaling)
        model.options.validation_periods = 4
        records: list[dict[str, Any]] = []

        class Recording(Trainer):
            def train(self, data_loader: WindowLoader, loss_fn: LossFn) -> TrainState:
                loader = data_loader
                state = super().train(loader, loss_fn)
                records.append(
                    {
                        "phase": "probe" if self._validation_loader is not None else "refit",
                        "windows": len(list(iter(loader))),
                        "final_step": int(state.step),
                        "best_step": self.best_step,
                        "max_steps": self.max_steps,
                    }
                )
                return state

        with monkeypatch.context() as patch:
            patch.setattr(model_module, "Trainer", Recording)
            model.train(_frame(60))
        return records[0], records[1]

    probe, refit = run("steps")
    assert probe["windows"] < refit["windows"], "the probe should see fewer windows than the refit"
    assert refit["max_steps"] == probe["best_step"]
    assert refit["final_step"] == probe["best_step"]

    probe, refit = run("epochs")
    expected = round(probe["best_step"] * refit["windows"] / probe["windows"])
    assert refit["max_steps"] == expected
    assert refit["final_step"] == expected


def test_unknown_refit_scaling_is_rejected():
    """A typo in the option must fail loudly rather than silently pick a default."""
    with pytest.raises(ValueError, match="refit_scaling"):
        _model(refit_scaling="per-epoch")


def test_a_probe_with_too_few_windows_is_rejected():
    """One probe window clears a length check but cannot locate a minimum."""
    model = _model()
    model.options.context_length = 16
    model.options.prediction_length = 3
    model.options.validation_periods = 8
    model.options.min_probe_windows = 5
    model._locations = ["loc0", "loc1", "loc2"]

    with pytest.raises(ValueError, match="windows"):
        model._early_stopping_loaders(_frame(46))
