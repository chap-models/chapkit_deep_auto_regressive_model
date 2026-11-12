"""Train and predict through the train and predict modules on the bundled weekly and monthly data.

Fast pass (AR_N_ITER=30, one ensemble member). Parity with the v2 models is
checked separately by tools/compare_v2.py, which needs their checkouts.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
DATA = {
    "week": REPO / "example_data/weekly/training_data.csv",
    "month": REPO / "example_data/monthly/training_data.csv",
}
HORIZON = {"week": 12, "month": 3}


def _train_and_predict(tmp_path: Path, period: str, df: pd.DataFrame, covariates=(), **options) -> pd.DataFrame:
    env = {**os.environ, "AR_N_ITER": "30"}
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump(
            {"additional_continuous_covariates": list(covariates), "user_option_values": {"n_ensemble": 1, **options}}
        )
    )
    data, model, out = tmp_path / "train.csv", tmp_path / "model", tmp_path / "predictions.csv"
    df.to_csv(data, index=False)
    future = pd.concat(
        [
            sub.sort_values("time_period").tail(HORIZON[period]).drop(columns=["disease_cases"])
            for _, sub in df.groupby("location", sort=False)
        ],
        ignore_index=True,
    )
    future.to_csv(tmp_path / "future.csv", index=False)
    subprocess.run(
        [sys.executable, "-m", "chapkit_auto_regressive.train", str(data), str(model), "--config", str(config)],
        cwd=REPO,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "chapkit_auto_regressive.predict",
            str(model),
            str(data),
            str(tmp_path / "future.csv"),
            str(out),
        ],
        cwd=REPO,
        env=env,
        check=True,
    )
    return pd.read_csv(out)


def _assert_forecast(out: pd.DataFrame, df: pd.DataFrame, period: str) -> None:
    samples = [c for c in out.columns if c.startswith("sample_")]
    assert {"time_period", "location"} <= set(out.columns)
    assert samples and np.isfinite(out[samples].to_numpy()).all()
    assert len(out) == df["location"].nunique() * HORIZON[period]


@pytest.mark.parametrize("period", ["week", "month"])
def test_train_then_predict(tmp_path: Path, period: str) -> None:
    df = pd.read_csv(DATA[period])
    _assert_forecast(_train_and_predict(tmp_path, period, df), df, period)


@pytest.mark.parametrize("period", ["week", "month"])
def test_declared_additional_covariate_is_used(tmp_path: Path, period: str) -> None:
    df = pd.read_csv(DATA[period]).assign(relative_humidity=lambda d: np.random.RandomState(0).rand(len(d)) * 100)
    _assert_forecast(_train_and_predict(tmp_path, period, df, covariates=["relative_humidity"]), df, period)


def test_undeclared_covariate_column_is_not_used(tmp_path: Path) -> None:
    # Chap writes every dataset column into the training CSV. A column the config
    # does not declare must not reach the network, so predicting without it works.
    df = pd.read_csv(DATA["month"])
    train = df.assign(relative_humidity=np.random.RandomState(0).rand(len(df)) * 100)
    env = {**os.environ, "AR_N_ITER": "30"}
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"user_option_values": {"n_ensemble": 1}}))
    train.to_csv(tmp_path / "train.csv", index=False)
    df.to_csv(tmp_path / "historic.csv", index=False)
    df.groupby("location").tail(3).drop(columns=["disease_cases"]).to_csv(tmp_path / "future.csv", index=False)
    cmd = [
        sys.executable,
        "-m",
        "chapkit_auto_regressive.train",
        str(tmp_path / "train.csv"),
        str(tmp_path / "m"),
        "--config",
    ]
    subprocess.run([*cmd, str(tmp_path / "config.yml")], cwd=REPO, env=env, check=True)
    files = [str(tmp_path / name) for name in ("m", "historic.csv", "future.csv", "out.csv")]
    subprocess.run([sys.executable, "-m", "chapkit_auto_regressive.predict", *files], cwd=REPO, env=env, check=True)
    assert len(pd.read_csv(tmp_path / "out.csv")) == df["location"].nunique() * 3


def test_weekly_context_is_clamped_to_short_history(tmp_path: Path) -> None:
    # 40 weeks is too short for the 52-week context plus the 12-week horizon, so
    # train.py shrinks the context instead of failing.
    df = pd.read_csv(DATA["week"])
    short = df.loc[df["time_period"].isin(sorted(df["time_period"].unique())[-40:])].reset_index(drop=True)
    _assert_forecast(_train_and_predict(tmp_path, "week", short), short, "week")


def test_warmup_cosine_schedule_trains(tmp_path: Path) -> None:
    df = pd.read_csv(DATA["week"])
    out = _train_and_predict(tmp_path, "week", df, lr_schedule="warmup_cosine", warmup_epochs=2)
    _assert_forecast(out, df, "week")
