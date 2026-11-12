"""Train entry point: ``python -m chapkit_auto_regressive.train <train_data.csv> <model_out> [--config config.yml]``.

The config is a Chap model-configuration YAML (``additional_continuous_covariates``
and ``prediction_periods`` at the top level, the model knobs under
``user_option_values``). chapkit writes it in that layout because main.py runs
the scripts with ``config_format="chap_core"`` (service.py), and chap-core writes the same
layout for the MLproject entry point.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel

from chapkit_auto_regressive.core import AutoRegressiveModel
from chapkit_auto_regressive.model import build_model, detect_period_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunConfig(BaseModel):
    """The parts of a Chap model-configuration YAML that training reads; other keys are ignored."""

    additional_continuous_covariates: list[str] | None = None
    prediction_periods: int | None = None
    user_option_values: dict[str, Any] | None = None


def _read_config(config_path: str | None) -> RunConfig:
    """Read and validate a Chap model-configuration YAML, if given."""
    if not config_path or not Path(config_path).exists():
        return RunConfig()
    with open(config_path) as f:
        return RunConfig.model_validate(yaml.safe_load(f) or {})


def _clamp_context_length(model: AutoRegressiveModel, data: pd.DataFrame) -> None:
    """Shrink ``context_length`` so the training data yields at least one window.

    Training needs ``context_length + prediction_length`` periods per window and
    prediction needs ``context_length`` periods of history, so the largest safe
    context is ``shortest history - prediction_length``. The clamped value is
    persisted in the trained predictor, so predict stays consistent.
    """
    period_counts: list[int] = data.groupby("location")["time_period"].nunique().tolist()
    periods = min(period_counts)
    max_context = periods - model.prediction_length
    if max_context < 1:
        raise ValueError(
            f"training data is too short: the shortest location history has {periods} periods, "
            f"but at least prediction_length+1={model.prediction_length + 1} are needed"
        )
    if model.context_length > max_context:
        logger.warning(
            "context_length=%d exceeds the available history (%d periods); clamping to %d",
            model.context_length,
            periods,
            max_context,
        )
        model.context_length = max_context


def main() -> None:
    """Train the model on a CSV and save the predictor."""
    parser = argparse.ArgumentParser(description="Train the deep auto-regressive model")
    parser.add_argument("train_data", help="path to the training data CSV")
    parser.add_argument("model", help="path to write the trained model")
    parser.add_argument("--config", default=None, help="path to a Chap model-configuration YAML")
    args = parser.parse_args()

    data = pd.read_csv(args.train_data)
    config = _read_config(args.config)
    period_type = detect_period_type(data["time_period"])
    model = build_model(config.user_option_values or {}, prediction_periods=config.prediction_periods)
    logger.info(
        "Period type %s: context_length=%d, prediction_length=%d, learning_rate=%g, n_iter=%d",
        period_type,
        model.context_length,
        model.prediction_length,
        model.learning_rate,
        model.n_iter,
    )
    _clamp_context_length(model, data)
    # Only the covariates the configuration declares are fed to the network on
    # top of the required three. Chap writes every dataset column into the
    # training CSV, so the columns present do not decide what is used. The chosen
    # covariates are persisted in the saved predictor, so predict.py needs no
    # matching configuration.
    model.additional_covariates = list(config.additional_continuous_covariates or [])
    if model.additional_covariates:
        logger.info("Using additional covariates: %s", model.additional_covariates)
    predictor = model.train(data)
    predictor.save(args.model)


if __name__ == "__main__":
    main()
