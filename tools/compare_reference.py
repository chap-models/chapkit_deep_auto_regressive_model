"""Compare this model's forecasts with a reference model (auto_regressive_{weekly,monthly}_v2).

Runs the reference repo's own train.py/predict.py (in its own locked environment,
via ``uv run``) and this repo's train/predict modules on the same data and the same
model configuration, then compares the prediction CSVs value by value.

Usage:
    uv run python tools/compare_reference.py --period week --reference ../auto_regressive_weekly_v2
    uv run python tools/compare_reference.py --period month --reference ../auto_regressive_monthly_v2

By default every case runs with ``AR_N_ITER=30`` to keep it fast (the code path
is the same at any epoch count); ``--full`` drops the override so the defaults
(1000 epochs weekly, 400 monthly) are compared as shipped. Exits 1 on any mismatch,
except that every case may differ by rounding (samples at most one case apart,
no forecast moving by more than 1%), and the cases in ``SHIFT_TOLERANCE`` may
differ, in short runs on the platforms where they are known to, within a bound
on how far each forecast's mean and its 10th and 90th percentiles move.
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from chapkit_deep_auto_regressive.model import WEEKLY_OPTIONS

#: Options every case starts from, per period type: weekly cases use the
#: recommended weekly values, which are the weekly reference's defaults. Both
#: sides get the same options.
BASE_OPTIONS: dict[str, dict[str, Any]] = {"week": WEEKLY_OPTIONS, "month": {}}

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
#: Cases allowed to differ from the reference, as ``(median, max)`` bounds on how
#: far the forecasts move. jax 0.12 computes the warmup_cosine learning-rate
#: schedule with slightly different rounding from the jax 0.10 the references
#: pin, and on Linux x86_64 and macOS training amplifies that into a small
#: shift. The bounds sit below the shift a different random seed gives in the
#: same case (median 18%, max 75%). Every other case must match exactly.
#:
#: The exception is narrow: it applies only to short runs (it was measured at 30
#: epochs) and not on Linux arm64, where these cases still match exactly. At
#: full length most cases drift on jax 0.12 (see the README), so ``--full`` runs
#: fail on them; there is no tolerance for full length yet. Its limits: the shift depends on the processor (an emulated
#: x86_64 run reached a median of 13%), so the check can fail on some CI
#: runners, and a bound this close to the seed-to-seed shift would not catch a
#: small real regression in these two cases.
SHIFT_TOLERANCE: dict[tuple[str, str], tuple[float, float]] = {
    ("week", "warmup-cosine"): (0.15, 0.30),
    ("week", "warmup-cosine-early-stopping"): (0.15, 0.30),
}

#: Every case, on any platform and at any length, also passes when its samples
#: differ by at most this many cases and no forecast's mean or 10th or 90th
#: percentile moves by more than ``ROUNDING_SHIFT``. GitHub's x86_64 runners have
#: different processors, and on some of them a sampled count rounds to the
#: neighbouring integer.
ROUNDING_DIFF = 1.0
ROUNDING_SHIFT = 0.01


def _tolerance_applies(full: bool) -> bool:
    """Whether ``SHIFT_TOLERANCE`` applies here: short runs, not on Linux arm64."""
    linux_arm64 = platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}
    return not full and not linux_arm64


def _shift(old: NDArray[np.floating[Any]], new: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Per-row shift of ``new`` from ``old``: relative to ``old``, or in cases when ``|old|`` is below 1."""
    return np.abs(new - old) / np.maximum(np.abs(old), 1.0)


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

    The references run ``train.py``/``predict.py`` from their root; this repo
    runs the ``chapkit_deep_auto_regressive.train``/``.predict`` modules.
    """
    uv = ["uv", "run", "--frozen", "--project", str(repo), "python"]
    train, predict = (
        (["-m", "chapkit_deep_auto_regressive.train"], ["-m", "chapkit_deep_auto_regressive.predict"])
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
    """Run every case for one period type; return whether every case passed.

    A case passes when its forecasts are identical, when they differ only by
    rounding (``ROUNDING_DIFF``, ``ROUNDING_SHIFT``), or when it is in
    ``SHIFT_TOLERANCE``, the tolerance applies here, and the forecasts move
    within its bounds.
    """
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
            config = {
                "additional_continuous_covariates": covariates,
                "user_option_values": {**BASE_OPTIONS[period], **options},
            }
            (workdir / "config.yml").write_text(yaml.safe_dump(config))

            old = _run(reference, workdir, env, "reference")
            new = _run(REPO, workdir, env, "this model")

        same_shape = list(old.columns) == list(new.columns) and old.shape == new.shape
        same_keys = same_shape and old[["time_period", "location"]].equals(new[["time_period", "location"]])
        samples = [c for c in old.columns if c.startswith("sample_")]
        max_diff = float(np.abs(old[samples].to_numpy() - new[samples].to_numpy()).max()) if same_keys else float("nan")
        identical = same_keys and old.equals(new)
        passed = identical
        status = "IDENTICAL" if identical else "DIFFERENT"
        detail = ""
        if same_keys and not identical:
            # One sampled draw can move far on a tiny change in the fitted
            # distribution, so compare each forecast's mean and its 10th and 90th
            # percentiles (the spread) rather than single draws.
            old_samples, new_samples = old[samples].to_numpy(dtype=float), new[samples].to_numpy(dtype=float)
            shifts = {
                "mean": _shift(old_samples.mean(axis=1), new_samples.mean(axis=1)),
                "p10": _shift(np.percentile(old_samples, 10, axis=1), np.percentile(new_samples, 10, axis=1)),
                "p90": _shift(np.percentile(old_samples, 90, axis=1), np.percentile(new_samples, 90, axis=1)),
            }
            median_shift = max(float(np.median(v)) for v in shifts.values())
            max_shift = max(float(v.max()) for v in shifts.values())
            detail = (
                " shift ("
                + ", ".join(f"{k} median {np.median(v):.1%} max {v.max():.1%}" for k, v in shifts.items())
                + ")"
            )
            tolerance = SHIFT_TOLERANCE.get((period, name))
            if max_diff <= ROUNDING_DIFF and max_shift <= ROUNDING_SHIFT:
                passed = True
                status = "ROUNDING"
            elif tolerance is not None and _tolerance_applies(full):
                passed = median_shift <= tolerance[0] and max_shift <= tolerance[1]
                status = "WITHIN TOLERANCE" if passed else "OUTSIDE TOLERANCE"
                detail += f" (allowed: median {tolerance[0]:.0%}, max {tolerance[1]:.0%})"
        ok &= passed
        print(
            f"[{period}] {name:<24} {status}  rows={len(new)} samples={len(samples)} max|diff|={max_diff:.3g}{detail}"
        )
        sys.stdout.flush()
    return ok


def main() -> None:
    """Parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--period", choices=["week", "month"], required=True)
    parser.add_argument("--reference", type=Path, required=True, help="checkout of the reference repo to compare with")
    parser.add_argument("--full", action="store_true", help="use the shipped epoch counts (slow)")
    parser.add_argument("--case", action="append", help="run only this case (repeatable)")
    args = parser.parse_args()
    if not compare(args.period, args.reference.resolve(), args.full, args.case):
        sys.exit(1)


if __name__ == "__main__":
    main()
