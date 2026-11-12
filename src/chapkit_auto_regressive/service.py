"""Chapkit service for the deep auto-regressive (RNN) model, for weekly and monthly data.

Merges ``auto_regressive_weekly_v2`` and ``auto_regressive_monthly_v2`` into one
model, with the shared ``chap_auto_regressive`` library vendored in as
``chapkit_auto_regressive.core``. With the same options and the same data, the
forecasts match the two models it replaces.

Design notes:

- The period type is not configured. The train module reads it from the training
  data's ``time_period`` column, and it picks the defaults for ``context_length``,
  ``prediction_length``, ``learning_rate`` and ``n_iter``. Those four fields
  default to 0, meaning "the period default", which keeps every option a flat
  scalar that Chap and the Modeling App render as a form control. See the open
  question in the README about how these defaults show in Chap.
- The train and predict modules are the same commands the MLproject runs. The
  runner writes config.yml in chap-core's layout (``config_format="chap_core"``),
  so they read ``user_option_values`` whichever way they are started.
- The MLproject is generated from this module (through the root main.py, which
  re-exports it): ``make mlproject``.
"""

import os
from pathlib import Path
from typing import Literal

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner
from pydantic import Field


class AutoRegressiveConfig(BaseConfig):
    """Tunable knobs of the deep auto-regressive model, as exposed to Chap and the Modeling App."""

    # ---- Period-dependent (0 = the default for the data's period type) ----
    context_length: int = Field(
        default=0,
        ge=0,
        description="Periods of history read as context. 0 uses the period default: 52 weeks or 36 months.",
    )
    prediction_length: int = Field(
        default=0,
        ge=0,
        description=(
            "Forecast horizon the model is trained for, which caps how far it can predict. 0 uses the period "
            "default (12 weeks or 3 months), raised to the requested prediction_periods when that is longer."
        ),
    )
    learning_rate: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Adam learning rate (with a schedule, its start and end rate). "
            "0 uses the period default: 1e-5 for weekly data, 1e-3 for monthly."
        ),
    )
    n_iter: int = Field(
        default=0,
        ge=0,
        description="Training epochs per ensemble member. 0 uses the period default: 1000 weekly, 400 monthly.",
    )

    # ---- Training ----
    n_ensemble: int = Field(default=5, description="Deep-ensemble members (seed-varied; pooled at predict time).")
    seed_offset: int = Field(default=0, description="Offset added to every ensemble member's seed (for repeat runs).")
    lr_schedule: Literal["constant", "warmup_cosine"] = Field(
        default="constant",
        description=(
            "Learning-rate schedule. 'warmup_cosine' warms up to peak_learning_rate, then cosine-decays to "
            "learning_rate over n_iter."
        ),
    )
    peak_learning_rate: float = Field(default=0.001, description="Peak learning rate of the warmup_cosine schedule.")
    warmup_epochs: int = Field(default=5, description="Linear warmup epochs of the warmup_cosine schedule.")

    # ---- Early stopping (opt-in) ----
    early_stopping: bool = Field(
        default=False,
        description=(
            "Stop each member at its best validation epoch, then refit on all data "
            "(needs a long series; see validation_periods)."
        ),
    )
    validation_periods: int = Field(default=12, description="Trailing periods held out to pick the stopping epoch.")
    min_probe_windows: int = Field(
        default=5, description="Training windows the early-stopping probe needs before the held-out block."
    )
    refit_scaling: Literal["epochs", "steps"] = Field(
        default="epochs", description="How the probe's stopping point carries over to the full refit."
    )
    patience: int = Field(default=6, description="Evaluations without improvement before stopping.")
    eval_every: int = Field(default=5, description="Epochs between validation evaluations.")

    # ---- Architecture (persisted in the trained model) ----
    cell: Literal["gru", "simple"] = Field(default="gru", description="Recurrent cell type.")
    rnn_features: int = Field(default=16, description="Hidden width of the encoder/decoder cells.")
    head_features: int = Field(default=24, description="Hidden width of the output head.")
    rnn_layers: int = Field(default=1, description="Number of stacked encoder/decoder RNN layers.")
    recursive_decode: bool = Field(
        default=False, description="Feed each forecast step's predicted mean back as the next step's input."
    )
    dropout_rate: float = Field(default=0.2, description="Dropout probability in the preprocess stage.")
    input_dropout_rate: float = Field(default=0.0, description="Feature-dropout probability on the input covariates.")


runner: ShellModelRunner[AutoRegressiveConfig] = ShellModelRunner(
    train_command="python -m chapkit_auto_regressive.train {data_file} model --config config.yml",
    predict_command="python -m chapkit_auto_regressive.predict model {historic_file} {future_file} {output_file}",
    config_format="chap_core",
)

info = MLServiceInfo(
    id="chapkit-auto-regressive",
    display_name="Deep Auto Regressive v2 (chapkit)",
    version="0.1.0",
    description=(
        "An experimental deep learning model based on an RNN architecture, focusing on predictions based on "
        "auto-regressive time series data. Supports weekly and monthly data: the period type is read from the "
        "data and picks the context, horizon and training defaults. Merges the Weekly and Monthly Deep Auto "
        "Regressive v2 models."
    ),
    model_metadata=ModelMetadata(
        author="Knut Rand",
        author_assessed_status=AssessedStatus.red,
        contact_email="knut.rand@dhis2.org",
        organization="HISP Centre, University of Oslo",
        organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
        citation_info=(
            'Rand, Knut. 2025. "Deep Auto Regressive model". HISP Centre, University of Oslo. '
            "https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html"
        ),
    ),
    period_type=PeriodType.any,
    allow_free_additional_continuous_covariates=True,
    required_covariates=["rainfall", "mean_temperature", "population"],
    min_prediction_periods=1,
    max_prediction_periods=12,
)

HIERARCHY = ArtifactHierarchy(
    name="chapkit_auto_regressive",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
if DATABASE_URL.startswith("sqlite") and ":///" in DATABASE_URL:
    db_path = Path(DATABASE_URL.split("///")[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=AutoRegressiveConfig,
        hierarchy=HIERARCHY,
        runner=runner,
        database_url=DATABASE_URL,
    )
    # Chap Core registration, configured by env (SERVICEKIT_ORCHESTRATOR_URL,
    # SERVICEKIT_REGISTRATION_KEY). Skipped when unset.
    .with_monitoring()
    .with_registration()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("chapkit_auto_regressive.service:app", reload=False)
