"""Compare this model's forecasts with the auto_regressive_{weekly,monthly}_v2 model it replaces.

Runs the reference repo's own train.py/predict.py (in its own locked environment,
via ``uv run``) and this repo's train/predict modules on the same data and the same
model configuration, then compares the prediction CSVs value by value.

Usage:
    uv run python tools/compare_v2.py --period week --reference ../auto_regressive_weekly_v2
    uv run python tools/compare_v2.py --period month --reference ../auto_regressive_monthly_v2

By default every case runs with ``AR_N_ITER=30`` to keep it fast (the code path
is the same at any epoch count); ``--full`` drops the override so the defaults
(1000 epochs weekly, 400 monthly) are compared as shipped. Exits 1 on any mismatch.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent

# Each case: (name, additional covariates, user_option_values, how to cut the data).
CASES: dict[str, list[tuple[str, list[str], dict[str, Any], str]]] = {
    "week": [
        ("defaults", [], {}, "all"),
        ("single-member", [], {"n_ensemble": 1}, "all"),
        ("additional-covariate", ["relative_humidity"], {"n_ensemble": 2}, "all"),
        ("simple-cell-2-layers", [], {"n_ensemble": 1, "cell": "simple", "rnn_layers": 2}, "all"),
        ("warmup-cosine", [], {"n_ensemble": 1, "lr_schedule": "warmup_cosine", "warmup_epochs": 2}, "all"),
        (
            "warmup-cosine-early-stopping",
            [],
            {"n_ensemble": 1, "lr_schedule": "warmup_cosine", "early_stopping": True},
            "all",
        ),
        ("short-history-clamp", [], {"n_ensemble": 1}, "last40"),
    ],
    "month": [
        ("defaults", [], {}, "all"),
        ("context-12", [], {"context_length": 12, "n_ensemble": 1}, "all"),
        ("additional-covariate", ["relative_humidity"], {"context_length": 12, "n_ensemble": 2}, "all"),
        (
            "regularised",
            [],
            {"context_length": 12, "n_ensemble": 1, "dropout_rate": 0.3, "input_dropout_rate": 0.25},
            "all",
        ),
        ("simple-cell-recursive", [], {"n_ensemble": 1, "cell": "simple", "recursive_decode": True}, "all"),
    ],
}
DATA = {
    "week": REPO / "example_data/weekly/training_data.csv",
    "month": REPO / "example_data/monthly/training_data.csv",
}
HORIZON = {"week": 12, "month": 3}


def _data(period: str, cut: str, with_covariate: bool) -> pd.DataFrame:
    df = pd.read_csv(DATA[period])
    if cut == "last40":
        periods = sorted(df["time_period"].unique())[-40:]
        df = df.loc[df["time_period"].isin(periods)].reset_index(drop=True)
    if with_covariate:
        df["relative_humidity"] = np.random.RandomState(0).rand(len(df)) * 100
    return df


def _run(repo: Path, workdir: Path, env: dict[str, str], tag: str) -> pd.DataFrame:
    """Train and predict with ``repo``'s entry points in its own uv environment.

    The v2 references run ``train.py``/``predict.py`` from their root; this repo
    runs the ``chapkit_auto_regressive.train``/``.predict`` modules.
    """
    uv = ["uv", "run", "--frozen", "--project", str(repo), "python"]
    train, predict = (
        (["-m", "chapkit_auto_regressive.train"], ["-m", "chapkit_auto_regressive.predict"])
        if repo == REPO
        else (["train.py"], ["predict.py"])
    )
    model = workdir / f"{tag}.model"
    out = workdir / f"{tag}.csv"
    subprocess.run(
        [*uv, *train, str(workdir / "train.csv"), str(model), "--config", str(workdir / "config.yml")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [*uv, *predict, str(model), str(workdir / "historic.csv"), str(workdir / "future.csv"), str(out)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    return pd.read_csv(out)


def compare(period: str, reference: Path, full: bool, only: list[str] | None) -> bool:
    """Run every case for one period type; return whether all matched exactly."""
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "AR_N_ITER"}}
    if not full:
        env["AR_N_ITER"] = "30"
    ok = True
    for name, covariates, options, cut in CASES[period]:
        if only and name not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            df = _data(period, cut, bool(covariates))
            df.to_csv(workdir / "train.csv", index=False)
            df.to_csv(workdir / "historic.csv", index=False)
            future = pd.concat(
                [
                    sub.sort_values("time_period").tail(HORIZON[period]).drop(columns=["disease_cases"])
                    for _, sub in df.groupby("location", sort=False)
                ],
                ignore_index=True,
            )
            future.to_csv(workdir / "future.csv", index=False)
            config = {"additional_continuous_covariates": covariates, "user_option_values": options}
            (workdir / "config.yml").write_text(yaml.safe_dump(config))

            old = _run(reference, workdir, env, "reference")
            new = _run(REPO, workdir, env, "merged")

        same_shape = list(old.columns) == list(new.columns) and old.shape == new.shape
        same_keys = same_shape and old[["time_period", "location"]].equals(new[["time_period", "location"]])
        samples = [c for c in old.columns if c.startswith("sample_")]
        max_diff = float(np.abs(old[samples].to_numpy() - new[samples].to_numpy()).max()) if same_keys else float("nan")
        identical = same_keys and old.equals(new)
        ok &= identical
        status = "IDENTICAL" if identical else "DIFFERENT"
        print(f"[{period}] {name:<24} {status}  rows={len(new)} samples={len(samples)} max|diff|={max_diff:.3g}")
        sys.stdout.flush()
    return ok


def main() -> None:
    """Parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--period", choices=["week", "month"], required=True)
    parser.add_argument("--reference", type=Path, required=True, help="checkout of the v2 repo to compare with")
    parser.add_argument("--full", action="store_true", help="use the shipped epoch counts (slow)")
    parser.add_argument("--case", action="append", help="run only this case (repeatable)")
    args = parser.parse_args()
    if not compare(args.period, args.reference.resolve(), args.full, args.case):
        sys.exit(1)


if __name__ == "__main__":
    main()
