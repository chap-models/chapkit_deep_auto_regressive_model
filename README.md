# chapkit_auto_regressive

Deep Auto Regressive model for Chap, as a [chapkit](https://github.com/dhis2-chap/chapkit)
service. An experimental deep learning model based on an RNN architecture (flax/jax), focusing
on predictions based on auto-regressive time series data.

It merges two models into one that handles **both weekly and monthly data**:

- [auto_regressive_weekly_v2](https://github.com/chap-models/auto_regressive_weekly_v2)
- [auto_regressive_monthly_v2](https://github.com/chap-models/auto_regressive_monthly_v2)

The shared model code, [chap_auto_regressive](https://github.com/chap-models/chap_auto_regressive),
is vendored in as `chapkit_auto_regressive.core` (from commit `c4d1d5c`). It is unchanged apart
from the package name and one addition: ISO-week periods (`2020W05`, `2020-W05`) parse, so the
model also accepts the week format chapkit uses. The v2 models only read `start/end` ranges.

Given the same options and data, the forecasts are identical to the two v2 models'
(see [Parity with the v2 models](#parity-with-the-v2-models)).

## Weekly and monthly data

The period type is not configured. It is read from the training data's `time_period` column
(`YYYY-MM` is monthly; a `start/end` date range or an ISO week, `YYYYWww` / `YYYY-Www`, is weekly), and it picks the defaults
for the four knobs that depend on it. These are the v2 models' defaults:

| option | weekly | monthly |
| --- | --- | --- |
| `context_length` (periods of history read) | 52 | 36 |
| `prediction_length` (trained horizon) | 12 | 3 |
| `learning_rate` | 1e-5 | 1e-3 |
| `n_iter` (epochs per ensemble member) | 1000 | 400 |

Each of these defaults to `0`, which means "use the default for the data's period type". Any
other value is used as given. When `prediction_length` is left on `0` and a run asks for a longer
horizon (`prediction_periods`) than the period default, the trained horizon is raised to match.
The v2 models would have rejected that run instead.

Everything else has one default for both period types: GRU cells, a 5-member deep ensemble,
dropout 0.2, a constant learning rate, and early stopping off. The weekly v2 model's optional
`warmup_cosine` learning-rate schedule is now part of the vendored library, so monthly runs can
use it too.

Covariates: `rainfall`, `mean_temperature` and `population` are required. Covariates named in
`additional_continuous_covariates` are fed to the network as extra features. Other columns in the
data are ignored. The architecture, window lengths and covariates are saved with the trained
model, so predict needs no configuration.

## Open questions

**How should the period-dependent defaults show in Chap?** The weekly and monthly v2 models use
different defaults for `context_length`, `prediction_length`, `learning_rate` and `n_iter`. The
service config and the MLproject list each option once with a single default, and Chap shows
that default to users. To keep forecasts the same as today, the merged model has to choose the
defaults from the data's period type. As a result, the default Chap shows is not the value that
actually runs.

The current choice is the `0` = "period default" sentinel above. It keeps every option a flat
scalar that Chap and the Modeling App render as a form control, and the descriptions name both
real defaults. The cost is that users see `0` rather than a number. Alternatives:

- Nullable options (`null` = period default): clearer in the schema, but chap-core's legacy
  option handling assumes one scalar `type` per option, and the Modeling App may fall back to
  raw-JSON editing.
- Weekly and monthly as two separately registered services from this one codebase, each with
  real defaults: what Chap shows matches what runs, but there are two models to register again.
- Pick one set of defaults: simplest, but the forecasts for the other period type change.

This needs a decision before the config is finalised.

## Layout

```
src/chapkit_auto_regressive/
  core/        vendored chap_auto_regressive library (network, trainer, data loading)
  model.py     period detection and period-aware model building
  train.py     train entry point:   python -m chapkit_auto_regressive.train <data> <model> --config <yml>
  predict.py   predict entry point: python -m chapkit_auto_regressive.predict <model> <historic> <future> <out>
  service.py   the chapkit service: config schema, ShellModelRunner, service info
main.py        re-exports the service at the root (uvicorn main:app, MLproject export)
MLproject      generated from main.py by `chapkit mlproject export` -- do not edit by hand
tools/         parity comparisons with the v2 models (script and API level)
configs/       example model configurations (weekly example, monthly full_rich)
scripts/       data preparation for the monthly full_rich configuration
example_data/  the v2 models' bundled weekly (Nicaragua) and monthly (Laos) data
```

This is a `uv_build` src-layout package. `uv build` (`make build-package`) builds the wheel, and
the Docker image installs it (non-editable) into the chapkit-py base image's venv.

## Running

```bash
make install       # uv sync
make test          # unit, library and train/predict tests (fast epochs)
make run           # build the image and serve on :8000
```

Without Docker: `uv run uvicorn main:app --port 8000`.

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

## Parity with the v2 models

`tools/compare_v2.py` runs a v2 repo's own `train.py`/`predict.py` (in that repo's locked
environment) and this model's modules on the same data and configuration, and requires the
prediction CSVs to be identical. `tools/api_compare.py` does the same through the service API.

```bash
make compare       # both period types, fast epochs (REF_WEEKLY / REF_MONTHLY point at the v2 checkouts)
make compare-full  # at the shipped epoch counts
make test-api      # through the Docker image's API
```

Cases cover the defaults, single and multi-member ensembles, an additional covariate,
architecture variants (simple cell, stacked layers, recursive decoding), regularisation, the
`warmup_cosine` schedule (weekly) and short-history context clamping (weekly). CI runs
`make compare` against both v2 repos' `main`.

Identical output depends on the numerical stack. `uv.lock` pins the same versions the v2 models
lock (jax/jaxlib 0.10.1, flax 0.12.7, optax 0.2.8, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1).
Upgrading them may change forecasts slightly, for both this model and the v2 models.

Differences from the v2 models that do not affect matching runs:

- The library classes moved package (`chap_auto_regressive` to `chapkit_auto_regressive.core`),
  and saved models pickle the scaler's class path, so v2 model files do not load here. Retrain.
- With `lr_schedule: warmup_cosine` **and** `early_stopping: true` together, the vendored
  library sizes the probe's schedule by the probe's own windows. The weekly v2 model's subclass
  sized it by the full series. All other combinations are unchanged.
- The weekly v2 `scripts/export_weights.py` (export for the Rust inference port) and
  `scripts/verify_eval.py` are not carried over.

## Configurations

- [`configs/weekly_example.yaml`](configs/weekly_example.yaml): starting point for weekly runs.
- [`configs/monthly_full_rich.yaml`](configs/monthly_full_rich.yaml): the tuned monthly
  configuration that beat the MSTL champion on the Rwanda dataset. Its feature CSV is built by
  [`scripts/data_prep/`](scripts/data_prep/).

`AR_N_ITER` overrides the epoch count regardless of configuration. The tests and the service
tests use it (`AR_N_ITER=30`) for a fast pass.

## Credits

Model by Knut Rand, HISP Centre, University of Oslo. AGPLv3, as the v2 models.
