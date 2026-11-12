"""Build the deep auto-regressive model for weekly or monthly data.

One model serves both period types. The period type is read from the training
data's ``time_period`` column, and it picks the defaults for the four knobs that
depend on it (context, horizon, learning rate, epochs). Everything else shares
one default. The values reproduce the two models this one replaces:

- ``auto_regressive_weekly_v2``: 52-week context, 12-week horizon, Adam at 1e-5
  for 1000 epochs.
- ``auto_regressive_monthly_v2``: 36-month context, 3-month horizon, Adam at
  1e-3 for 400 epochs (the library defaults).
"""

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from chapkit_auto_regressive.core import AutoRegressiveModel

logger = logging.getLogger(__name__)

PeriodType = Literal["week", "month"]

#: Knobs whose default depends on the period type. A run that sets one of them to
#: ``AUTO`` (or leaves it out) gets the default for its data's period type.
PERIOD_DEFAULTS: dict[PeriodType, dict[str, Any]] = {
    "week": {"context_length": 52, "prediction_length": 12, "learning_rate": 1e-5, "n_iter": 1000},
    "month": {"context_length": 36, "prediction_length": 3, "learning_rate": 1e-3, "n_iter": 400},
}

#: Sentinel for "use the period type's default". None of the period-dependent
#: knobs has a meaningful zero, and a plain number keeps every option a flat
#: scalar that chap and the Modeling App can render as a form control.
AUTO = 0

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
    period_type: PeriodType,
    options: Mapping[str, Any] | None = None,
    prediction_periods: int | None = None,
) -> dict[str, Any]:
    """Return the run's model options with the period defaults filled in.

    Args:
        period_type: ``"week"`` or ``"month"``.
        options: A run's ``user_option_values``. A period-dependent knob that is
            missing, ``None`` or ``AUTO`` takes the period default.
        prediction_periods: The horizon the run asks for. When the horizon the
            model is trained for is left on auto, it is raised to this if the
            period default is shorter, so the forecast the run asks for can be
            made at all (the trained horizon caps how far predict can reach).
    """
    resolved = {key: value for key, value in (options or {}).items() if value is not None}
    for key, default in PERIOD_DEFAULTS[period_type].items():
        if resolved.get(key, AUTO) == AUTO:
            resolved[key] = default
            if key == "prediction_length" and prediction_periods and prediction_periods > default:
                resolved[key] = prediction_periods
    return resolved


def build_model(
    period_type: PeriodType,
    options: Mapping[str, Any] | None = None,
    prediction_periods: int | None = None,
) -> AutoRegressiveModel:
    """Return the model for a period type, applying any run ``user_option_values``.

    Unknown option keys are ignored with a warning. ``AR_N_ITER`` overrides the
    epoch count regardless of config, so the tests can run a fast pass.
    """
    resolved = resolve_options(period_type, options, prediction_periods)
    if "AR_N_ITER" in os.environ:
        resolved["n_iter"] = int(os.environ["AR_N_ITER"])
    model = AutoRegressiveModel()
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
