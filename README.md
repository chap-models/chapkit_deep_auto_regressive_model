# chapkit_deep_auto_regressive_model

Deep Auto Regressive: a deep learning model for Chap that forecasts disease cases from their own
history and climate covariates, for **weekly and monthly data**. It is served with
[chapkit](https://github.com/dhis2-chap/chapkit).

## The model

An encoder-decoder recurrent neural network (flax/jax, GRU cells by default). For each location it
reads a window of past cases and covariates, adds a learned per-location embedding and seasonal
calendar features, and outputs a negative binomial distribution for every future period. Five
networks trained with different seeds form a deep ensemble, and their samples are pooled into the
forecast.

Covariates: `rainfall`, `mean_temperature` and `population` are required. Covariates named in
`additional_continuous_covariates` are fed to the network as extra features. Other columns in the
data are ignored. The architecture, window lengths and covariates are saved with the trained
model, so predict needs no configuration.

## Weekly and monthly data

The period type is not configured. Weekly (`start/end` date range or ISO week, `YYYYWww` /
`YYYY-Www`) and monthly (`YYYY-MM`) data are both accepted and checked when training starts.
Each location's periods are ordered by date, so ISO weeks (`2020W2`, `2020W10`, or both ISO forms
mixed) stay in calendar order. One set of defaults serves both. For weekly data, four options have recommended values of their own:

| option | default | recommended for weekly data |
| --- | --- | --- |
| `context_length` (periods of history read) | 36 | 52 |
| `prediction_length` (trained horizon) | 3 | 12 |
| `learning_rate` | 1e-3 | 1e-5 |
| `n_iter` (epochs per ensemble member) | 400 | 1000 |

[`configs/weekly_example.yaml`](configs/weekly_example.yaml) sets the weekly values, and
`chapkit_deep_auto_regressive.model.WEEKLY_OPTIONS` holds them.

When a run asks for a longer horizon (`prediction_periods`) than `prediction_length`, the trained
horizon is raised to match, because predict cannot reach past it. This is what lets a weekly run
with the defaults ask Chap for 12 weeks.

Other defaults: GRU cells, a 5-member deep ensemble, dropout 0.2, a constant learning rate
(`lr_schedule: warmup_cosine` warms up to a peak and decays), and early stopping off.

## Known issues

**One-period forecasts crash with recursive decoding.** With `recursive_decode: true`, a
prediction for a single future period fails with `ValueError: Need at least one array to stack`
(`core/rnn_model.py:226`). The decoder loop gets an empty sequence, so `jnp.stack(etas)` has
nothing to stack. The service advertises one-period forecasts (`min_prediction_periods` 1), so
this is reachable from Chap. Fixing it means handling the empty decoder sequence and adding a
regression test. Until then, leave `recursive_decode` off (the default) or request two or more
periods.

## Layout

```
src/chapkit_deep_auto_regressive/
  core/        the network, training loop and data handling
  model.py     defaults, option handling and period detection
  train.py     train entry point:   python -m chapkit_deep_auto_regressive.train <data> <model> --config <yml>
  predict.py   predict entry point: python -m chapkit_deep_auto_regressive.predict <model> <historic> <future> <out>
  service.py   the chapkit service: config schema, ShellModelRunner, service info
main.py        re-exports the service at the root (uvicorn main:app, MLproject export)
MLproject      generated from main.py by `chapkit mlproject export` -- do not edit by hand
tools/         reference forecast comparisons (script and API level) and the benchmark
configs/       example model configurations (weekly example, monthly full_rich)
scripts/       data preparation for the monthly full_rich configuration
example_data/  example weekly (Nicaragua) and monthly (Laos) data
```

This is a `uv_build` src-layout package. `uv build` (`make build-package`) builds the wheel, and
the Docker image installs it (non-editable) into the chapkit-py base image's venv.

## Running

```bash
make install       # uv sync
make lint          # ruff (incl. docstring and security rules), mypy, pyright (strict), MLproject check
make test          # unit, core and train/predict tests (fast epochs)
make run           # build the image and serve on :8000
```

Without Docker: `uv run uvicorn main:app --port 8000`.

Linting and type checking follow chapkit's setup (ruff `E,W,F,I,D,S`, mypy, pyright in strict
mode) for the whole package and its tests.

The service uses `ShellModelRunner` with `config_format="chap_core"`. Each train or predict job
runs the train or predict module in its own workspace, with a `config.yml` in chap-core's layout:
`additional_continuous_covariates` and `prediction_periods` at the top level, and the knobs under
`user_option_values`.

### MLproject

The MLproject is generated from the service, never edited by hand:

```bash
make mlproject     # chapkit mlproject export
make lint          # includes chapkit mlproject export --check
```

`chapkit mlproject run .` serves the MLproject as the same service (`make test-mlproject`).

### Evaluating with the Chap CLI

Run the model as a chapkit service and point `chap eval` at it, either a running service or the
Docker image:

```bash
make run           # in one terminal
chap eval \
  --model-name http://localhost:8000 --run-config.is-chapkit-model \
  --dataset-csv example_data/weekly/training_data.csv \
  --model-configuration-yaml configs/weekly_example.yaml \
  --backtest-params.n-splits 3 --output-file eval.nc

chap eval \
  --model-name docker://ghcr.io/chap-models/chapkit_deep_auto_regressive_model:latest \
  --dataset-csv example_data/monthly/training_data.csv \
  --backtest-params.n-splits 3 --output-file eval.nc
```

`--model-name` with the repository itself (a local path or the GitHub URL), with or without
`--run-config.as-chapkit`, does not work yet: chap-core runs the commands with its own Python, in
which this package is not installed, and chapkit's MLproject export has no option to declare
an environment (`uv_env` / `docker_env`).

## Reference check

With the same options and data, the forecasts are identical to those of the two models this one
replaces, [auto_regressive_weekly_v2](https://github.com/chap-models/auto_regressive_weekly_v2)
(with the weekly values above) and
[auto_regressive_monthly_v2](https://github.com/chap-models/auto_regressive_monthly_v2) (with the
defaults), with one exception, described below. `tools/compare_reference.py` runs a reference
repo's own scripts in its locked environment and this model's modules on the same data, and
compares the prediction CSVs; `tools/api_compare.py` does the same through the service API. CI
runs `make compare`.

The exception: with `lr_schedule: warmup_cosine` the weekly forecasts differ on some processors,
because jax 0.12 computes that schedule with different rounding from the jax 0.10 the reference
models pin. On Linux arm64 they are identical; on macOS arm64 samples differ by up to one case;
on Linux x86_64 the forecast means move by a median of 3.4% (at most 12%) in the 30-epoch CI
comparison. A different random seed moves them far more (median 18%, at most 75%).

So in short runs, except on Linux arm64, `tools/compare_reference.py` lets the two weekly
warmup-cosine cases differ as long as each forecast's mean and its 10th and 90th percentiles move
by a median of at most 15% and at most 30% in any one forecast. The shift is relative to the
reference value, or in cases when that is below 1. Every other case, every case on Linux arm64,
and every `--full` run must match exactly.

The tolerance has two limits. The shift depends on the processor running the check, and one
emulated x86_64 run reached a median of 13%, close to the bound, so the check can fail on some CI
runners. The 10th percentile has the least room: on GitHub CI it moved by up to 25% in one forecast,
against the 30% bound. And because the bound is near the seed-to-seed shift, it would not catch a
small real regression in these two cases.

### Rolling back the dependency upgrade

The upgrade to jax 0.12
([#2](https://github.com/chap-models/chapkit_deep_auto_regressive_model/pull/2)) is the only
reason the reference check needs a tolerance. To go back to exact parity with the reference
models:

1. Revert the commits of #2 on a branch (`git revert` of each, newest first) and open a pull
   request. That restores jax/jaxlib 0.10.1, flax 0.12.7, optax 0.2.8, numpy 2.4.6, pandas 3.0.3
   and scipy 1.17.1 in `pyproject.toml` and `uv.lock`, the previous `chapkit-py` digest, and the
   exact reference check.
2. Until that is merged, run the image built before the upgrade: the Docker workflow tags every
   build of `main` as `sha-<short commit>`, so
   `ghcr.io/chap-models/chapkit_deep_auto_regressive_model:sha-e1f765f` is the last one on jax
   0.10.1.

```bash
make compare       # fast epochs (REF_WEEKLY / REF_MONTHLY point at the reference checkouts)
make compare-full  # full epoch counts
make test-api      # through the Docker image's API
```

Identical output depends on the numerical stack, which `uv.lock` pins (jax/jaxlib 0.12.0.1, flax
0.12.12, optax 0.2.9, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1), and on the platform. The same
model on Linux (the Docker image) and on macOS gives forecasts that differ by up to one case in
some samples over a full-length training run, with the same mean; compare like with like.

## Benchmark

`tools/benchmark.py` (`make bench`) trains and predicts on the weekly and monthly example data
and reports the median time and peak memory of each step. It saves the forecasts, so a run before
and a run after a change can be checked for identical output:

```bash
uv run python tools/benchmark.py --out /tmp/before          # on the base branch
uv run python tools/benchmark.py --out /tmp/after --compare /tmp/before
```

It uses 100 epochs by default (`--n-iter`); training time grows linearly with the epoch count.

## Configurations

- [`configs/weekly_example.yaml`](configs/weekly_example.yaml): starting point for weekly runs.
- [`configs/monthly_full_rich.yaml`](configs/monthly_full_rich.yaml): the tuned monthly
  configuration that beat the MSTL champion on the Rwanda dataset. Its feature CSV is built by
  [`scripts/data_prep/`](scripts/data_prep/).

`AR_N_ITER` overrides the epoch count regardless of configuration. The tests and the service
tests use it (`AR_N_ITER=30`) for a fast pass.


## Credits

Model by Knut Rand, HISP Centre, University of Oslo. AGPLv3.
