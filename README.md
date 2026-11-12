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
`YYYY-Www`) and monthly (`YYYY-MM`) data are both accepted and checked when training starts. One set
of defaults serves both. For weekly data, four options have recommended values of their own:

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
  core/        vendored chap_auto_regressive library (network, trainer, data loading)
  model.py     defaults, option handling and period detection
  train.py     train entry point:   python -m chapkit_deep_auto_regressive.train <data> <model> --config <yml>
  predict.py   predict entry point: python -m chapkit_deep_auto_regressive.predict <model> <historic> <future> <out>
  service.py   the chapkit service: config schema, ShellModelRunner, service info
main.py        re-exports the service at the root (uvicorn main:app, MLproject export)
MLproject      generated from main.py by `chapkit mlproject export` -- do not edit by hand
tools/         reference forecast comparisons (script and API level)
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
make test          # unit, library and train/predict tests (fast epochs)
make run           # build the image and serve on :8000
```

Without Docker: `uv run uvicorn main:app --port 8000`.

Linting and type checking follow chapkit's setup (ruff `E,W,F,I,D,S`, mypy, pyright in strict
mode), except for the vendored `core/` library and its tests in `tests/library/`. Those get
ruff's style and import rules only, so the copy stays close to upstream; their typing is a
matter for `chap_auto_regressive`.

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

The exported MLproject does not run directly under chap-core's own MLproject runner
(`chap eval --model-name .`). That runner writes the configuration to
`model_configuration_for_run.yaml` and passes it as `{model_config}`, whereas chapkit's exporter
writes the literal `config.yml` the chapkit runner uses. The exporter also emits no environment
hint (`uv_env` / `docker_env`), so the commands would run in chap-core's own environment. Serve
the MLproject with chapkit, as the GHR model does.

## Vendored library

The network, trainer and data loading come from
[chap_auto_regressive](https://github.com/chap-models/chap_auto_regressive), vendored in as
`chapkit_deep_auto_regressive.core` (from commit `c4d1d5c`). Apart from the package name it has
three local changes:

- ISO-week periods (`2020W05`, `2020-W05`) parse, the week format chapkit uses. Upstream reads
  only `start/end` ranges.
- Each location's periods are ordered by start date rather than as text, so ISO weeks such as
  `2020W2` and `2020W10`, or the two ISO forms mixed, stay in calendar order.
- With `lr_schedule: warmup_cosine` and `early_stopping: true` together, the early-stopping
  probe's schedule is sized by the full series. Upstream sizes it by the probe's own windows.

## Reference check

With the same options and data, the forecasts are identical to those of the two models this one
replaces, [auto_regressive_weekly_v2](https://github.com/chap-models/auto_regressive_weekly_v2)
(with the weekly values above) and
[auto_regressive_monthly_v2](https://github.com/chap-models/auto_regressive_monthly_v2) (with the
defaults). `tools/compare_reference.py` runs a reference repo's own scripts in its locked
environment and this model's modules on the same data, and requires identical prediction CSVs;
`tools/api_compare.py` does the same through the service API. CI runs `make compare`.

```bash
make compare       # fast epochs (REF_WEEKLY / REF_MONTHLY point at the reference checkouts)
make compare-full  # full epoch counts
make test-api      # through the Docker image's API
```

Identical output depends on the numerical stack, which `uv.lock` pins (jax/jaxlib 0.10.1, flax
0.12.7, optax 0.2.8, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1).

## Configurations

- [`configs/weekly_example.yaml`](configs/weekly_example.yaml): starting point for weekly runs.
- [`configs/monthly_full_rich.yaml`](configs/monthly_full_rich.yaml): the tuned monthly
  configuration that beat the MSTL champion on the Rwanda dataset. Its feature CSV is built by
  [`scripts/data_prep/`](scripts/data_prep/).

`AR_N_ITER` overrides the epoch count regardless of configuration. The tests and the service
tests use it (`AR_N_ITER=30`) for a fast pass.


## Credits

Model by Knut Rand, HISP Centre, University of Oslo. AGPLv3.
