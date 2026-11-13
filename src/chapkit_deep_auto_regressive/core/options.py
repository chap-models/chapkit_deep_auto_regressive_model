"""The model's options, in one place.

[`TrainingOptions`][chapkit_deep_auto_regressive.core.options.TrainingOptions] holds every knob a Chap run
may set; the service config is built from it, so a default or a description is
written once. [`ModelOptions`][chapkit_deep_auto_regressive.core.options.ModelOptions] adds the network
widths and covariates that are fixed per deployment, and
[`Architecture`][chapkit_deep_auto_regressive.core.options.Architecture] is the part of them that is saved
with a trained model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CellType = Literal["gru", "simple"]
LearningRateSchedule = Literal["constant", "warmup_cosine"]
RefitScaling = Literal["epochs", "steps"]


class TrainingOptions(BaseModel):
    """The knobs a Chap run may set (its ``user_option_values``)."""

    # ---- Context, horizon and training length (recommended weekly values in the descriptions) ----
    context_length: int = Field(
        default=36,
        gt=0,
        description="Periods of history read as context. For weekly data, 52 is recommended.",
    )
    prediction_length: int = Field(
        default=3,
        gt=0,
        description=(
            "Forecast horizon the model is trained for, which caps how far it can predict. Raised to the requested "
            "prediction_periods when that is longer. For weekly data, 12 is recommended."
        ),
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Adam learning rate (with a schedule, its start and end rate). For weekly data, 1e-5 is recommended."
        ),
    )
    n_iter: int = Field(
        default=400,
        gt=0,
        description="Training epochs per ensemble member. For weekly data, 1000 is recommended.",
    )

    # ---- Training ----
    n_ensemble: int = Field(default=5, ge=1, description="Deep-ensemble members (seed-varied; pooled at predict time).")
    seed_offset: int = Field(default=0, description="Offset added to every ensemble member's seed (for repeat runs).")
    lr_schedule: LearningRateSchedule = Field(
        default="constant",
        description=(
            "Learning-rate schedule. 'warmup_cosine' warms up to peak_learning_rate, then cosine-decays to "
            "learning_rate over n_iter."
        ),
    )
    peak_learning_rate: float = Field(
        default=0.001, gt=0.0, description="Peak learning rate of the warmup_cosine schedule."
    )
    warmup_epochs: int = Field(default=5, ge=0, description="Linear warmup epochs of the warmup_cosine schedule.")

    # ---- Early stopping (opt-in) ----
    early_stopping: bool = Field(
        default=False,
        description=(
            "Stop each member at its best validation epoch, then refit on all data "
            "(needs a long series; see validation_periods)."
        ),
    )
    validation_periods: int = Field(
        default=12, gt=0, description="Trailing periods held out to pick the stopping epoch."
    )
    min_probe_windows: int = Field(
        default=5, gt=0, description="Training windows the early-stopping probe needs before the held-out block."
    )
    refit_scaling: RefitScaling = Field(
        default="epochs", description="How the probe's stopping point carries over to the full refit."
    )
    patience: int = Field(default=6, gt=0, description="Evaluations without improvement before stopping.")
    eval_every: int = Field(default=5, gt=0, description="Epochs between validation evaluations.")

    # ---- Architecture (persisted in the trained model) ----
    cell: CellType = Field(default="gru", description="Recurrent cell type.")
    rnn_features: int = Field(default=16, gt=0, description="Hidden width of the encoder/decoder cells.")
    head_features: int = Field(default=24, gt=0, description="Hidden width of the output head.")
    rnn_layers: int = Field(default=1, gt=0, description="Number of stacked encoder/decoder RNN layers.")
    recursive_decode: bool = Field(
        default=False, description="Feed each forecast step's predicted mean back as the next step's input."
    )
    dropout_rate: float = Field(default=0.2, ge=0.0, lt=1.0, description="Dropout probability in the preprocess stage.")
    input_dropout_rate: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Feature-dropout probability on the input covariates."
    )


class Architecture(BaseModel):
    """The network hyperparameters, saved with a trained model so predict rebuilds the same network."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell: CellType
    rnn_features: int
    preprocess_hidden: int
    preprocess_output: int
    embedding_dim: int
    head_features: int
    rnn_layers: int
    recursive_decode: bool
    dropout_rate: float
    input_dropout_rate: float


class ModelOptions(TrainingOptions):
    """Every option of the model: the run knobs plus the fixed network widths and covariates."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    preprocess_hidden: int = Field(default=16, gt=0, description="Hidden width of the per-location preprocess stack.")
    preprocess_output: int = Field(default=8, gt=0, description="Output width of the preprocess stage.")
    embedding_dim: int = Field(default=8, gt=0, description="Size of the per-location embedding.")
    additional_covariates: tuple[str, ...] = Field(
        default=(),
        description=(
            "Extra covariate columns used as features, after the required rainfall, mean_temperature and population."
        ),
    )

    @property
    def architecture(self) -> Architecture:
        """The network hyperparameters of these options."""
        return Architecture.model_validate(self.model_dump(include=set(Architecture.model_fields)))
