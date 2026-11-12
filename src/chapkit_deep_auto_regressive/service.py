"""Chapkit service for the deep auto-regressive (RNN) model, for weekly and monthly data.

The network, training loop and data handling are in ``chapkit_deep_auto_regressive.core``.

Design notes:

- One set of defaults serves both period types. ``context_length``,
  ``prediction_length``, ``learning_rate`` and ``n_iter`` have recommended
  weekly values (``model.WEEKLY_OPTIONS``; see configs/weekly_example.yaml).
- The train and predict modules are the same commands the MLproject runs. The
  runner writes config.yml in chap-core's layout (``config_format="chap_core"``),
  so they read ``user_option_values`` whichever way they are started.
- The MLproject is generated from this module (through the root main.py, which
  re-exports it): ``make mlproject``.
"""

import os
from pathlib import Path

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner
from pydantic import HttpUrl

from chapkit_deep_auto_regressive.core.options import TrainingOptions


class AutoRegressiveConfig(BaseConfig, TrainingOptions):
    """Tunable knobs of the deep auto-regressive model, as exposed to Chap and the Modeling App.

    The knobs, their defaults and descriptions are those of ``TrainingOptions``.
    """


runner: ShellModelRunner[AutoRegressiveConfig] = ShellModelRunner(
    train_command="python -m chapkit_deep_auto_regressive.train {data_file} model --config config.yml",
    predict_command="python -m chapkit_deep_auto_regressive.predict model {historic_file} {future_file} {output_file}",
    config_format="chap_core",
)

info = MLServiceInfo(
    id="deep-auto-regressive",
    display_name="Deep Auto Regressive (weekly and monthly)",
    version="0.1.0",
    description=(
        "An experimental deep learning model: an ensemble of recurrent neural networks that forecasts disease "
        "cases from their own history and climate covariates. Supports weekly and monthly data. For weekly "
        "data, context_length 52, prediction_length 12, learning_rate 1e-5 and n_iter 1000 are recommended."
    ),
    model_metadata=ModelMetadata(
        author="Knut Rand",
        author_assessed_status=AssessedStatus.red,
        contact_email="knut.rand@dhis2.org",
        organization="HISP Centre, University of Oslo",
        organization_logo_url=HttpUrl("https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png"),
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
    name="chapkit_deep_auto_regressive",
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

    run_app("chapkit_deep_auto_regressive.service:app", reload=False)
