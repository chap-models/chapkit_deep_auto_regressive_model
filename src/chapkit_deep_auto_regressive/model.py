"""Build the deep auto-regressive model for weekly or monthly data.

One model serves both period types, with one set of defaults (36-period
context, 3-period horizon, Adam at 1e-3 for 400 epochs, the library defaults).
Weekly data has recommended values of its own for those four knobs,
``WEEKLY_OPTIONS`` (as ``configs/weekly_example.yaml`` sets).
"""

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from chapkit_deep_auto_regressive.core import DeepAutoRegressiveModel

logger = logging.getLogger(__name__)

PeriodType = Literal["week", "month"]

#: The defaults of the knobs that have recommended weekly values (the library's defaults).
DEFAULTS: dict[str, Any] = {"context_length": 36, "prediction_length": 3, "learning_rate": 1e-3, "n_iter": 400}

#: Recommended values of those knobs for weekly data.
WEEKLY_OPTIONS: dict[str, Any] = {
    "context_length": 52,
    "prediction_length": 12,
    "learning_rate": 1e-5,
    "n_iter": 1000,
}

#: Model options a run may set via ``user_option_values``. The service config in
#: service.py declares exactly these (tests/test_periods.py checks it).
MODEL_OPTIONS = (
    "n_iter",
    "seed_offset",
    "early_stopping",
    "validation_periods",
    "patience",
    "eval_every",
    "min_probe_windows",
    "refit_scaling",
    "context_length",
    "n_ensemble",
    "learning_rate",
    "lr_schedule",
    "peak_learning_rate",
    "warmup_epochs",
    "prediction_length",
    "cell",
    "rnn_features",
    "head_features",
    "rnn_layers",
    "dropout_rate",
    "recursive_decode",
    "input_dropout_rate",
)

# chap-core sends monthly periods as "YYYY-MM", and weekly ones either as an ISO
# week ("2020W05", "2020-W05") or as a "start/end" date range.
_MONTHLY = re.compile(r"^\d{4}-\d{2}$")
_WEEKLY = re.compile(r"^(\d{4}-?W\d{1,2}|\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2})$")


def detect_period_type(time_periods: Iterable[Any]) -> PeriodType:
    """Return whether the periods are weekly or monthly.

    Raises:
        ValueError: If there are no periods, a period matches neither format, or
            the periods mix the two.
    """
    kinds: set[PeriodType] = set()
    for value in time_periods:
        period = str(value)
        if _MONTHLY.match(period):
            kinds.add("month")
        elif _WEEKLY.match(period):
            kinds.add("week")
        else:
            raise ValueError(f"time_period {period!r} is neither monthly (YYYY-MM) nor weekly (YYYYWww or start/end)")
    if len(kinds) != 1:
        raise ValueError("time_period is empty" if not kinds else "time_period mixes weekly and monthly periods")
    return kinds.pop()


def resolve_options(
    options: Mapping[str, Any] | None = None,
    prediction_periods: int | None = None,
) -> dict[str, Any]:
    """Return the run's model options with the defaults filled in.

    Args:
        options: A run's ``user_option_values``. A knob that is missing or
            ``None`` takes its default.
        prediction_periods: The horizon the run asks for. The trained horizon
            caps how far predict can reach, so ``prediction_length`` is raised to
            this when it is shorter.
    """
    resolved = {**DEFAULTS, **{key: value for key, value in (options or {}).items() if value is not None}}
    if prediction_periods and prediction_periods > resolved["prediction_length"]:
        logger.info(
            "prediction_length=%d is shorter than the %d periods asked for; raising it to %d",
            resolved["prediction_length"],
            prediction_periods,
            prediction_periods,
        )
        resolved["prediction_length"] = prediction_periods
    return resolved


def build_model(
    options: Mapping[str, Any] | None = None,
    prediction_periods: int | None = None,
) -> DeepAutoRegressiveModel:
    """Return the model with the defaults and any run ``user_option_values`` applied.

    Unknown option keys are ignored with a warning. ``AR_N_ITER`` overrides the
    epoch count regardless of config, so the tests can run a fast pass.
    """
    resolved = resolve_options(options, prediction_periods)
    if "AR_N_ITER" in os.environ:
        resolved["n_iter"] = int(os.environ["AR_N_ITER"])
    model = DeepAutoRegressiveModel()
    # Early stopping is opt-in. It holds out a trailing validation block and needs
    # a long series (the library raises when the history is too short), so a run
    # must ask for it explicitly rather than inherit it from the library default.
    model.early_stopping = False
    for key, value in resolved.items():
        if key in MODEL_OPTIONS:
            setattr(model, key, value)
        else:
            logger.warning("Ignoring unknown model option: %s", key)
    return model
