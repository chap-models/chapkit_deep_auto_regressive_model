"""Compare forecasts made through the chapkit service API with the reference model.

Drives a running service the way Chap does (POST config, POST $train, poll the
job, POST $predict, read the prediction artifact) on the bundled example data,
and runs the reference repo's own scripts on the same data and options. The
two forecasts must be identical.

Start the service with the same AR_N_ITER this script uses (30 unless --full),
for example ``AR_N_ITER=30 uv run uvicorn main:app --port 8131``, then:

    uv run python tools/api_compare.py --url http://localhost:8131 \
        --period week --reference ../auto_regressive_weekly_v2
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pandas as pd
import yaml

# Run as a script (python tools/api_compare.py), so its own folder is on sys.path.
from compare_reference import BASE_OPTIONS, DATA, HORIZON, _run

# (name, user options) per period type; options go to the service config and the reference config alike.
CASES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "week": [("defaults", {}), ("simple-cell", {"n_ensemble": 1, "cell": "simple"})],
    "month": [("defaults", {}), ("context-12", {"n_ensemble": 1, "context_length": 12})],
}


def _frame(df: pd.DataFrame) -> dict[str, Any]:
    """Return a pandas frame in chapkit's {columns, data} shape."""
    return {"columns": list(df.columns), "data": df.astype(object).where(df.notna(), None).values.tolist()}


def _wait(client: httpx2.Client, job_id: str, label: str) -> None:
    """Poll a job until it finishes; raise unless it completed."""
    while True:
        job = client.get(f"/api/v1/jobs/{job_id}").raise_for_status().json()
        if job["status"] in ("completed", "failed", "cancelled"):
            if job["status"] != "completed":
                raise RuntimeError(f"{label} {job['status']}: {str(job)[:1500]}")
            return
        time.sleep(2)


def _service_forecast(
    client: httpx2.Client, name: str, options: dict[str, Any], train: pd.DataFrame, future: pd.DataFrame
) -> pd.DataFrame:
    """Train and predict through the API; return the forecast frame."""
    config = client.post("/api/v1/configs", json={"name": f"api-compare-{name}", "data": options})
    config_id = config.raise_for_status().json()["id"]
    train_job = client.post("/api/v1/ml/$train", json={"config_id": config_id, "data": _frame(train)})
    train_job = train_job.raise_for_status().json()
    _wait(client, train_job["job_id"], "train")
    payload = {"artifact_id": train_job["artifact_id"], "historic": _frame(train), "future": _frame(future)}
    predict_job = client.post("/api/v1/ml/$predict", json=payload).raise_for_status().json()
    _wait(client, predict_job["job_id"], "predict")
    artifact = client.get(f"/api/v1/artifacts/{predict_job['artifact_id']}").raise_for_status().json()
    content = artifact["data"]["content"]
    return pd.DataFrame(content["data"], columns=content["columns"])


def main() -> None:
    """Run every case for one period type against the service and the reference."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--period", choices=["week", "month"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="the service runs without AR_N_ITER")
    args = parser.parse_args()

    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "AR_N_ITER"}}
    if not args.full:
        env["AR_N_ITER"] = "30"
    train = pd.read_csv(DATA[args.period])
    future = pd.concat(
        [
            sub.sort_values("time_period").tail(HORIZON[args.period]).drop(columns=["disease_cases"])
            for _, sub in train.groupby("location", sort=False)
        ],
        ignore_index=True,
    )
    ok = True
    with httpx2.Client(base_url=args.url, timeout=120) as client:
        for name, case_options in CASES[args.period]:
            options = {**BASE_OPTIONS[args.period], **case_options}
            service = _service_forecast(client, name, options, train, future)
            with tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                train.to_csv(workdir / "train.csv", index=False)
                train.to_csv(workdir / "historic.csv", index=False)
                future.to_csv(workdir / "future.csv", index=False)
                config: dict[str, Any] = {"additional_continuous_covariates": [], "user_option_values": options}
                (workdir / "config.yml").write_text(yaml.safe_dump(config))
                reference = _run(args.reference.resolve(), workdir, env, "reference")
            samples = [c for c in reference.columns if c.startswith("sample_")]
            keys = ["time_period", "location"]
            same_keys = list(service.columns) == list(reference.columns) and service[keys].astype(str).equals(
                reference[keys].astype(str)
            )
            diff = np.abs(service[samples].to_numpy(float) - reference[samples].to_numpy(float)).max()
            identical = same_keys and diff == 0
            ok &= identical
            status = "IDENTICAL" if identical else "DIFFERENT"
            print(f"[api {args.period}] {name:<12} {status}  rows={len(service)} max|diff|={diff:.3g}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
