"""Unit tests for the period-aware model builder and the service config."""

import pytest

from chapkit_auto_regressive.model import (
    AUTO,
    MODEL_OPTIONS,
    PERIOD_DEFAULTS,
    build_model,
    detect_period_type,
    resolve_options,
)
from chapkit_auto_regressive.service import AutoRegressiveConfig


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        (["2010-01", "2010-02"], "month"),
        (["2019-10-14/2019-10-20", "2019-10-21/2019-10-27"], "week"),
        (["2020W05", "2020W06"], "week"),
        (["2020-W05"], "week"),
    ],
)
def test_detect_period_type(periods, expected):
    assert detect_period_type(periods) == expected


@pytest.mark.parametrize("periods", [[], ["2020-01-05"], ["2020", "2020-01"], ["2010-01", "2020W05"], ["2020SunW5"]])
def test_detect_period_type_rejects_unknown_or_mixed(periods):
    with pytest.raises(ValueError):
        detect_period_type(periods)


def test_weekly_defaults_match_auto_regressive_weekly_v2(monkeypatch):
    monkeypatch.delenv("AR_N_ITER", raising=False)
    model = build_model("week")
    assert (model.context_length, model.prediction_length, model.learning_rate, model.n_iter) == (52, 12, 1e-5, 1000)
    assert model.early_stopping is False


def test_monthly_defaults_match_auto_regressive_monthly_v2(monkeypatch):
    monkeypatch.delenv("AR_N_ITER", raising=False)
    model = build_model("month")
    assert (model.context_length, model.prediction_length, model.learning_rate, model.n_iter) == (36, 3, 1e-3, 400)
    assert model.early_stopping is False


def test_auto_and_none_take_the_period_default_and_explicit_values_win():
    options = {"context_length": AUTO, "learning_rate": None, "n_iter": 50, "prediction_length": 6}
    resolved = resolve_options("week", options)
    assert resolved["context_length"] == PERIOD_DEFAULTS["week"]["context_length"]
    assert resolved["learning_rate"] == PERIOD_DEFAULTS["week"]["learning_rate"]
    assert (resolved["n_iter"], resolved["prediction_length"]) == (50, 6)


def test_auto_horizon_is_raised_to_the_requested_prediction_periods():
    assert resolve_options("month", {}, prediction_periods=6)["prediction_length"] == 6
    assert resolve_options("month", {}, prediction_periods=2)["prediction_length"] == 3
    assert resolve_options("month", {"prediction_length": 2}, prediction_periods=6)["prediction_length"] == 2


def test_build_model_applies_user_options_and_ignores_unknown():
    model = build_model("month", {"context_length": 18, "n_ensemble": 3, "cell": "simple", "bogus_option": 1})
    assert (model.context_length, model.n_ensemble, model.cell) == (18, 3, "simple")
    assert not hasattr(model, "bogus_option")


def test_service_config_declares_every_model_option():
    # Chap only offers what the config (and the MLproject generated from it)
    # declares, so an option the code accepts but the config omits cannot be set.
    declared = set(AutoRegressiveConfig.model_fields) - {"prediction_periods", "additional_continuous_covariates"}
    assert declared == set(MODEL_OPTIONS)


def test_service_config_defaults_reproduce_the_v2_models(monkeypatch):
    # The config defaults, passed through as user_option_values, must build the
    # same model as the v2 defaults did.
    monkeypatch.delenv("AR_N_ITER", raising=False)
    options = AutoRegressiveConfig().model_dump(exclude={"prediction_periods", "additional_continuous_covariates"})
    for period in ("week", "month"):
        from_config, plain = build_model(period, options), build_model(period)
        assert {key: getattr(from_config, key) for key in MODEL_OPTIONS} == {
            key: getattr(plain, key) for key in MODEL_OPTIONS
        }


def test_iso_weeks_start_on_their_monday_and_other_formats_are_unchanged():
    # ISO weeks are a local addition to the vendored library; the formats the v2
    # models used must parse exactly as before.
    from datetime import datetime

    from chapkit_auto_regressive.core.transforms import period_start_date

    assert period_start_date("2020W01") == period_start_date("2020-W01") == datetime(2019, 12, 30)
    with pytest.raises(ValueError):
        period_start_date("2021-W53")  # 2021 has no ISO week 53
    assert period_start_date("2020-W53") == datetime(2020, 12, 28)
    assert period_start_date("2010-03") == datetime(2010, 3, 1)
    assert period_start_date("2019-10-14/2019-10-20") == datetime(2019, 10, 14)


@pytest.mark.parametrize(
    "periods",
    [
        ["2020W10", "2020W1", "2020W2", "2020W9"],  # unpadded: text order is 1, 10, 2, 9
        ["2020-W03", "2020W01", "2020-W02", "2020W04"],  # mixed ISO forms
        ["2020-W53", "2021W01", "2020W52"],  # across a year boundary
    ],
)
def test_iso_weeks_are_ordered_by_date_not_text(periods):
    import pandas as pd

    from chapkit_auto_regressive.core.transforms import location_groups, period_start_date

    frame = pd.DataFrame({"location": "a", "time_period": periods})
    ((_, rows),) = list(location_groups(frame))
    starts = [period_start_date(p) for p in rows["time_period"]]
    assert starts == sorted(starts)


def test_monthly_and_range_order_is_unchanged():
    import pandas as pd

    from chapkit_auto_regressive.core.transforms import location_groups

    for periods in (
        ["2020-03", "2019-12", "2020-01", "2020-02"],
        ["2019-10-21/2019-10-27", "2019-10-07/2019-10-13", "2019-10-14/2019-10-20"],
    ):
        frame = pd.DataFrame({"location": "a", "time_period": periods})
        ((_, rows),) = list(location_groups(frame))
        assert rows["time_period"].tolist() == sorted(periods)
