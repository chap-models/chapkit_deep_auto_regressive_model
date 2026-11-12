"""Time training and prediction, and keep the forecasts for a before/after check.

Runs this repo's train and predict modules on the bundled weekly and monthly
example data, a few times each, and reports the median wall time and the peak
memory of each step. The forecasts of the last run are written to ``--out`` so
two runs (say, before and after a change) can be checked for identical output
with ``--compare``.

Usage:
    uv run python tools/benchmark.py --out /tmp/before
    # ... change the code ...
    uv run python tools/benchmark.py --out /tmp/after --compare /tmp/before

Epochs default to 100 (``--n-iter``) so a run takes minutes rather than an
hour; training time scales linearly with the epoch count.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from chapkit_deep_auto_regressive.core.transforms import period_start_date
from chapkit_deep_auto_regressive.model import WEEKLY_OPTIONS

REPO = Path(__file__).resolve().parent.parent
#: (period, training data, forecast horizon, user_option_values)
CASES: list[tuple[str, Path, int, dict[str, Any]]] = [
    ("week", REPO / "example_data/weekly/training_data.csv", 12, dict(WEEKLY_OPTIONS)),
    ("month", REPO / "example_data/monthly/training_data.csv", 3, {}),
]


def _run(args: list[str], env: dict[str, str]) -> tuple[float, float]:
    """Run a command and return its wall time (s) and peak memory (MiB)."""
    # The log goes to a file, not a pipe: a pipe nobody reads while waiting for
    # the process would fill up on a long run and block it.
    with tempfile.TemporaryFile() as log:
        start = time.perf_counter()
        process = subprocess.Popen(args, env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=log)
        _, status, usage = os.wait4(process.pid, 0)
        elapsed = time.perf_counter() - start
        if status != 0:
            log.seek(0)
            raise RuntimeError(f"{' '.join(args)} failed:\n{log.read().decode()[-3000:]}")
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    peak = usage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else usage.ru_maxrss / 1024
    return elapsed, peak


def _case(period: str, data: Path, horizon: int, options: dict[str, Any], workdir: Path) -> None:
    """Split the data into history and future, and write the run config."""
    df = pd.read_csv(data)
    periods = sorted(df["time_period"].unique(), key=period_start_date)
    cut = set(periods[-horizon:])
    history = df.loc[~df["time_period"].isin(cut)]
    future = df.loc[df["time_period"].isin(cut)].drop(columns="disease_cases")
    history.to_csv(workdir / "train.csv", index=False)
    history.to_csv(workdir / "historic.csv", index=False)
    future.to_csv(workdir / "future.csv", index=False)
    config = {"prediction_periods": horizon, "user_option_values": options}
    (workdir / "config.yml").write_text(yaml.safe_dump(config))


def main() -> None:
    """Benchmark both period types and optionally compare with an earlier run."""
    parser = argparse.ArgumentParser(description="Time training and prediction, and keep the forecasts")
    parser.add_argument("--out", required=True, type=Path, help="directory for the results and forecasts")
    parser.add_argument("--compare", type=Path, help="an earlier --out directory to compare forecasts with")
    parser.add_argument("--n-iter", type=int, default=100, help="training epochs per ensemble member")
    parser.add_argument("--repeats", type=int, default=3, help="runs per case; the median time is reported")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "AR_N_ITER": str(args.n_iter)}
    python = [sys.executable, "-m"]
    results: dict[str, dict[str, float]] = {}
    for period, data, horizon, options in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _case(period, data, horizon, options, workdir)
            train_runs: list[tuple[float, float]] = []
            predict_runs: list[tuple[float, float]] = []
            for _ in range(args.repeats):
                train_runs.append(
                    _run(
                        [
                            *python,
                            "chapkit_deep_auto_regressive.train",
                            str(workdir / "train.csv"),
                            str(workdir / "model"),
                            "--config",
                            str(workdir / "config.yml"),
                        ],
                        env,
                    )
                )
                predict_runs.append(
                    _run(
                        [
                            *python,
                            "chapkit_deep_auto_regressive.predict",
                            str(workdir / "model"),
                            str(workdir / "historic.csv"),
                            str(workdir / "future.csv"),
                            str(args.out / f"{period}_forecast.csv"),
                        ],
                        env,
                    )
                )
            results[period] = {
                "train_s": statistics.median(t for t, _ in train_runs),
                "train_peak_mib": max(m for _, m in train_runs),
                "predict_s": statistics.median(t for t, _ in predict_runs),
                "predict_peak_mib": max(m for _, m in predict_runs),
            }
        r = results[period]
        print(
            f"[{period}] train {r['train_s']:.1f}s ({r['train_peak_mib']:.0f} MiB), "
            f"predict {r['predict_s']:.1f}s ({r['predict_peak_mib']:.0f} MiB)",
            flush=True,
        )
    (args.out / "timings.json").write_text(json.dumps({"n_iter": args.n_iter, "results": results}, indent=2))

    if args.compare is None:
        return
    before = json.loads((args.compare / "timings.json").read_text())["results"]
    identical = True
    for period, r in results.items():
        b = before[period]
        old = pd.read_csv(args.compare / f"{period}_forecast.csv")
        new = pd.read_csv(args.out / f"{period}_forecast.csv")
        samples = [c for c in old.columns if c.startswith("sample_")]
        same = old.shape == new.shape and np.array_equal(old[samples].to_numpy(), new[samples].to_numpy())
        identical &= same
        diff = "IDENTICAL" if same else "DIFFERENT"
        print(
            f"[{period}] forecasts {diff}; train {b['train_s']:.1f}s -> {r['train_s']:.1f}s "
            f"({r['train_s'] / b['train_s'] - 1:+.0%}), predict {b['predict_s']:.1f}s -> {r['predict_s']:.1f}s "
            f"({r['predict_s'] / b['predict_s'] - 1:+.0%}), peak train memory "
            f"{b['train_peak_mib']:.0f} -> {r['train_peak_mib']:.0f} MiB"
        )
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
