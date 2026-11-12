"""Unit tests for period handling, the model builder and the service config."""

from pathlib import Path

import pytest
import yaml

from chapkit_deep_auto_regressive.model import (
    DEFAULTS,
    MODEL_OPTIONS,
    WEEKLY_V2_OPTIONS,
    build_model,
    detect_period_type,
    resolve_options,
)
from chapkit_deep_auto_regressive.service import AutoRegressiveConfig


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        (["2010-01", "2010-02"], "month"),
        (["2019-10-14/2019-10-20", "2019-10-21/2019-10-27"], "week"),
        (["2020W05", "2020W06"], "week"),
        (["2020-W05"], "week"),
    ],
)
def test_detect_period_type(periods: list[str], expected: str) -> None:
    assert detect_period_type(periods) == expected


@pytest.mark.parametrize("periods", [[], ["2020-01-05"], ["2020", "2020-01"], ["2010-01", "2020W05"], ["2020SunW5"]])
def test_detect_period_type_rejects_unknown_or_mixed(periods: list[str]) -> None:
    with pytest.raises(ValueError):
        detect_period_type(periods)


def test_defaults_match_auto_regressive_monthly_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AR_N_ITER", raising=False)
    model = build_model()
    assert (model.context_length, model.prediction_length, model.learning_rate, model.n_iter) == (36, 3, 1e-3, 400)
    assert model.early_stopping is False


def test_weekly_v2_options_reproduce_auto_regressive_weekly_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AR_N_ITER", raising=False)
    model = build_model(WEEKLY_V2_OPTIONS)
    assert (model.context_length, model.prediction_length, model.learning_rate, model.n_iter) == (52, 12, 1e-5, 1000)


def test_weekly_example_config_sets_the_weekly_v2_values() -> None:
    config = yaml.safe_load((Path(__file__).parent.parent / "configs/weekly_example.yaml").read_text())
    assert {key: config["user_option_values"][key] for key in WEEKLY_V2_OPTIONS} == WEEKLY_V2_OPTIONS


def test_none_takes_the_default_and_explicit_values_win() -> None:
    resolved = resolve_options({"context_length": None, "n_iter": 50, "prediction_length": 6})
    assert resolved["context_length"] == DEFAULTS["context_length"]
    assert resolved["learning_rate"] == DEFAULTS["learning_rate"]
    assert (resolved["n_iter"], resolved["prediction_length"]) == (50, 6)


def test_horizon_is_raised_to_the_requested_prediction_periods() -> None:
    assert resolve_options({}, prediction_periods=6)["prediction_length"] == 6
    assert resolve_options({}, prediction_periods=2)["prediction_length"] == 3
    assert resolve_options({"prediction_length": 2}, prediction_periods=6)["prediction_length"] == 6
    assert resolve_options({"prediction_length": 12}, prediction_periods=6)["prediction_length"] == 12


def test_build_model_applies_user_options_and_ignores_unknown() -> None:
    model = build_model({"context_length": 18, "n_ensemble": 3, "cell": "simple", "bogus_option": 1})
    assert (model.context_length, model.n_ensemble, model.cell) == (18, 3, "simple")
    assert not hasattr(model, "bogus_option")


def test_service_config_declares_every_model_option() -> None:
    # Chap only offers what the config (and the MLproject generated from it)
    # declares, so an option the code accepts but the config omits cannot be set.
    declared = set(AutoRegressiveConfig.model_fields) - {"prediction_periods", "additional_continuous_covariates"}
    assert declared == set(MODEL_OPTIONS)


def test_service_config_defaults_are_the_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # The config defaults Chap shows, passed through as user_option_values, must
    # build the same model as no options at all (the monthly v2 model).
    monkeypatch.delenv("AR_N_ITER", raising=False)
    options = AutoRegressiveConfig().model_dump(exclude={"prediction_periods", "additional_continuous_covariates"})
    from_config, plain = build_model(options), build_model()
    assert {key: getattr(from_config, key) for key in MODEL_OPTIONS} == {
        key: getattr(plain, key) for key in MODEL_OPTIONS
    }


def test_iso_weeks_start_on_their_monday_and_other_formats_are_unchanged() -> None:
    # ISO weeks are a local addition to the vendored library; the formats the v2
    # models used must parse exactly as before.
    from datetime import datetime

    from chapkit_deep_auto_regressive.core.transforms import period_start_date

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
def test_iso_weeks_are_ordered_by_date_not_text(periods: list[str]) -> None:
    import pandas as pd

    from chapkit_deep_auto_regressive.core.transforms import location_groups, period_start_date

    frame = pd.DataFrame({"location": "a", "time_period": periods})
    ((_, rows),) = list(location_groups(frame))
    starts = [period_start_date(p) for p in rows["time_period"]]
    assert starts == sorted(starts)


def test_monthly_and_range_order_is_unchanged() -> None:
    import pandas as pd

    from chapkit_deep_auto_regressive.core.transforms import location_groups

    for periods in (
        ["2020-03", "2019-12", "2020-01", "2020-02"],
        ["2019-10-21/2019-10-27", "2019-10-07/2019-10-13", "2019-10-14/2019-10-20"],
    ):
        frame = pd.DataFrame({"location": "a", "time_period": periods})
        ((_, rows),) = list(location_groups(frame))
        assert rows["time_period"].tolist() == sorted(periods)


def test_run_config_reads_chap_layout_and_ignores_other_keys(tmp_path: Path) -> None:
    from chapkit_deep_auto_regressive.train import _read_config

    config = tmp_path / "config.yml"
    config.write_text(
        "additional_continuous_covariates: [relative_humidity]\n"
        "prediction_periods: 3\n"
        "user_option_values: {n_ensemble: 1}\n"
        "model_name: something chap also writes\n"
    )
    run = _read_config(str(config))
    assert run.additional_continuous_covariates == ["relative_humidity"]
    assert run.prediction_periods == 3
    assert run.user_option_values == {"n_ensemble": 1}
    assert _read_config(None).user_option_values is None
