"""Build the deep auto-regressive model for weekly or monthly data.

One model serves both period types, with one set of defaults (36-period
context, 3-period horizon, Adam at 1e-3 for 400 epochs; see
``core.options.TrainingOptions``). Weekly data has recommended values of its
own for those four knobs, ``WEEKLY_OPTIONS`` (as ``configs/weekly_example.yaml`` sets).
"""

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from chapkit_deep_auto_regressive.core import DeepAutoRegressiveModel
from chapkit_deep_auto_regressive.core.options import ModelOptions, TrainingOptions

logger = logging.getLogger(__name__)

PeriodType = Literal["week", "month"]

#: Recommended values for weekly data of the knobs whose defaults suit monthly data.
WEEKLY_OPTIONS: dict[str, Any] = {
    "context_length": 52,
    "prediction_length": 12,
    "learning_rate": 1e-5,
    "n_iter": 1000,
}

#: Model options a run may set via ``user_option_values``; the service config declares exactly these.
MODEL_OPTIONS: tuple[str, ...] = tuple(TrainingOptions.model_fields)

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
) -> ModelOptions:
    """Return the run's model options with the defaults filled in.

    Args:
        options: A run's ``user_option_values``. A knob that is missing or
            ``None`` takes its default; an unknown one is ignored with a warning.
        prediction_periods: The horizon the run asks for. The trained horizon
            caps how far predict can reach, so ``prediction_length`` is raised to
            this when it is shorter.

    Raises:
        pydantic.ValidationError: If a value is invalid, such as an unknown cell type.
    """
    values: dict[str, Any] = {}
    for key, value in (options or {}).items():
        if key not in MODEL_OPTIONS:
            logger.warning("Ignoring unknown model option: %s", key)
        elif value is not None:
            values[key] = value
    if "AR_N_ITER" in os.environ:
        # Lets the tests run a fast pass, whatever the config says.
        values["n_iter"] = int(os.environ["AR_N_ITER"])
    resolved = ModelOptions.model_validate(values)
    if prediction_periods and prediction_periods > resolved.prediction_length:
        logger.info(
            "prediction_length=%d is shorter than the %d periods asked for; raising it to %d",
            resolved.prediction_length,
            prediction_periods,
            prediction_periods,
        )
        resolved.prediction_length = prediction_periods
    return resolved


def build_model(
    options: Mapping[str, Any] | None = None,
    prediction_periods: int | None = None,
) -> DeepAutoRegressiveModel:
    """Return the model with the defaults and any run ``user_option_values`` applied."""
    return DeepAutoRegressiveModel(resolve_options(options, prediction_periods))
