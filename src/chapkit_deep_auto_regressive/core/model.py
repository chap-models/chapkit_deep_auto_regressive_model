"""The public model and its trained predictor.

[`DeepAutoRegressiveModel`][chapkit_deep_auto_regressive.core.model.DeepAutoRegressiveModel] is the estimator CHAP
trains; its ``train`` method returns a
[`FlaxPredictor`][chapkit_deep_auto_regressive.core.model.FlaxPredictor] that produces probabilistic
forecasts. Both share the same output head — the network emits two channels per
period (``eta``) which [`nb_head`][chapkit_deep_auto_regressive.core.distributions.nb_head] turns into a
negative-binomial distribution that is then sampled.

The public API speaks tidy :class:`pandas.DataFrame` objects (one row per location
and time period); the model itself has no dependency on chap-core.
"""

import logging
import pickle
from collections.abc import Sequence
from typing import Any, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .data_loader import DataSet as DLDataSet
from .data_loader import Features, SimpleDataLoader, Transform
from .distributions import nb_head
from .rnn_model import build_network
from .trainer import Trainer
from .transforms import REQUIRED_COVARIATES, ZScaler, get_series, location_groups


class Architecture(TypedDict):
    """The network hyperparameters, persisted with a trained predictor."""

    cell: str
    rnn_features: int
    preprocess_hidden: int
    preprocess_output: int
    embedding_dim: int
    head_features: int
    rnn_layers: int
    recursive_decode: bool
    dropout_rate: float
    input_dropout_rate: float


def _apply_network(model: nn.Module, params: Any, x: Features, y: ArrayLike) -> jax.Array:
    """Run the network forward and return its output ``eta``.

    ``nn.Module.apply`` also returns the updated state when ``mutable`` is set,
    which is never the case here.
    """
    eta: jax.Array | tuple[Any, Any] = model.apply(params, x, y)
    if isinstance(eta, tuple):
        raise TypeError("the network returned mutable state; expected its output only")
    return eta


logger = logging.getLogger(__name__)


def _check_predict_inputs(
    historic: pd.DataFrame,
    future: pd.DataFrame,
    prediction_length: int,
    context_length: int,
    training_locations: list[Any] | None = None,
) -> None:
    """Validate the history/future frames before forecasting.

    Args:
        historic: Recent history including observed cases.
        future: The periods to forecast, with covariates but no cases.
        prediction_length: The fixed forecast horizon the model was built for.
        context_length: Number of past periods the model reads as context; each
            location's history must provide at least this many.
        training_locations: The canonical (sorted) locations the model was trained
            on. When provided, the frames must cover exactly this set — the model
            learns per-location embeddings by sorted index, so an unseen location
            would silently borrow another location's embedding.

    Raises:
        ValueError: If the two frames cover different location sets, if the
            locations differ from ``training_locations``, if any location provides
            fewer than ``context_length`` history periods, or if any location asks
            for more than ``prediction_length`` future periods.
    """
    historic_locations = set(historic["location"].unique())
    if historic_locations != set(future["location"].unique()):
        raise ValueError("historic and future must cover the same set of locations")
    if training_locations is not None and sorted(historic_locations) != sorted(training_locations):
        raise ValueError(
            "prediction locations must match the training locations "
            f"{sorted(training_locations)}, but got {sorted(historic_locations)}"
        )
    historic_counts = historic.groupby("location").size()
    if (historic_counts < context_length).any():
        raise ValueError(
            f"each location's history must have at least context_length={context_length} periods, "
            f"got period counts {sorted(set(historic_counts.tolist()))}"
        )
    # The model forecasts up to its trained horizon; callers (e.g. a chap eval
    # backtest) may request fewer periods, but more than prediction_length would
    # extrapolate beyond what the network was trained for.
    counts = future.groupby("location").size()
    if (counts > prediction_length).any():
        raise ValueError(
            f"each location's future must have at most prediction_length={prediction_length} "
            f"periods, got period counts {sorted(set(counts.tolist()))}"
        )


def _forecast_frame(future: pd.DataFrame, samples: NDArray[Any]) -> pd.DataFrame:
    """Assemble the prediction output frame from per-location samples.

    Args:
        future: The future-covariate frame the forecast was made for.
        samples: Sampled counts with a leading location axis, each entry shaped
            ``(periods, n_samples)`` (aligned with
            [`location_groups`][chapkit_deep_auto_regressive.core.transforms.location_groups]).

    Returns:
        A long frame with columns ``time_period``, ``location`` and one
        ``sample_i`` column per draw.
    """
    groups = list(location_groups(future))
    frames: list[pd.DataFrame] = []
    for (location, sub), location_samples in zip(groups, samples):
        location_samples = np.asarray(location_samples)
        frame = pd.DataFrame({f"sample_{i}": location_samples[:, i] for i in range(location_samples.shape[-1])})
        frame.insert(0, "location", location)
        # Label each location with its own forecast periods. location_groups
        # sorts each group by time_period, matching the order the samples were
        # produced in, so locations with differing future periods stay correct.
        frame.insert(0, "time_period", sub["time_period"].to_numpy())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class FlaxPredictor:
    """A trained model that turns history and future covariates into samples.

    Produced by
    [`DeepAutoRegressiveModel.train`][chapkit_deep_auto_regressive.core.model.DeepAutoRegressiveModel.train].
    It holds the fitted network parameters and the feature scaler, and can be
    pickled to disk and reloaded.

    Attributes:
        distribution_head: Maps the network's two-channel output to a sampling
            distribution (the negative-binomial head).
    """

    distribution_head = staticmethod(nb_head)

    def __init__(
        self,
        params: Any,
        transform: Transform,
        architecture: Architecture,
        prediction_length: int,
        context_length: int,
        locations: list[Any] | None = None,
        covariates: Sequence[str] = REQUIRED_COVARIATES,
    ) -> None:
        # The network is rebuilt from the saved architecture spec, so a loaded
        # predictor is self-contained and does not depend on how the caller's
        # DeepAutoRegressiveModel happens to be configured.
        self.architecture: Architecture = architecture.copy()
        # Canonical (sorted) training locations; predictions must cover this exact
        # set because the network's per-location embeddings are indexed by sorted
        # position. None when constructed without them, which skips the check.
        self.locations = list(locations) if locations is not None else None
        self.model = build_network(len(self.locations) if self.locations else 1, **self.architecture)
        # Accept either a single parameter pytree or a list of them (a deep
        # ensemble). Stored as a list; predict pools the members' samples.
        self._params_list: list[Any] = params if isinstance(params, list) else [params]
        self._params = self._params_list[0]
        self._transform = transform
        self.prediction_length = prediction_length
        self.context_length = context_length
        # The covariate columns (in order) the network was trained on; predict
        # must build features the same way.
        self.covariates = tuple(covariates)
        self.rng_key = jax.random.PRNGKey(1234)

    def get_samples(self, eta: jax.Array, n_samples: int) -> NDArray[Any]:
        """Draw samples from the output distribution for each period.

        Args:
            eta: The network's per-period output, shape ``(..., periods, 2)``.
            n_samples: Number of samples to draw per period.

        Returns:
            Sampled counts, with a leading sample axis of length ``n_samples``.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))

    def predict(self, historic: pd.DataFrame, future: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
        """Forecast the future periods as samples per location.

        The last ``context_length`` periods of history (with observed cases) are
        concatenated with the future covariates, run through the network, and the
        forecast-horizon outputs are sampled.

        Args:
            historic: Recent history including observed ``disease_cases``.
            future: The periods to forecast, with covariates but no cases.
            num_samples: Number of samples to draw per location and period.

        Returns:
            A frame with columns ``time_period``, ``location`` and one ``sample_i``
            column per draw.
        """
        _check_predict_inputs(historic, future, self.prediction_length, self.context_length, self.locations)
        x, _ = get_series(future, self.covariates)
        prev_values, prev_y = get_series(historic, self.covariates)
        prev_values = prev_values[:, -self.context_length :]
        prev_y = prev_y[:, -self.context_length :]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        dataset.set_transform(self._transform)
        window_x, window_y = dataset.prediction_instance()
        n_prev = prev_values.shape[1]
        # Pool samples across the ensemble members. Each member draws an equal
        # share of num_samples (the first members absorb the remainder) so the
        # output keeps exactly num_samples columns.
        n_members = len(self._params_list)
        shares = [num_samples // n_members + (1 if k < num_samples % n_members else 0) for k in range(n_members)]
        member_samples: list[NDArray[Any]] = []
        for params, share in zip(self._params_list, shares):
            if share == 0:
                continue
            eta = _apply_network(self.model, params, window_x, window_y)
            member_samples.append(np.asarray(self.get_samples(eta[:, n_prev - 1 :], share)))
        samples = np.concatenate(member_samples, axis=-1)  # pool along the trailing sample axis
        return _forecast_frame(future, samples)

    def save(self, path: str) -> None:
        """Pickle everything needed to reload a self-contained predictor.

        The payload carries the ensemble parameters, the feature scaler, the
        training locations and covariates, **and the network architecture and
        window lengths** — so ``load`` rebuilds the exact network without needing
        the original model configuration.

        Args:
            path: Destination file path.
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params_list": self._params_list,
                    "transform": self._transform,
                    "locations": self.locations,
                    "covariates": self.covariates,
                    "architecture": self.architecture,
                    "prediction_length": self.prediction_length,
                    "context_length": self.context_length,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FlaxPredictor":
        """Reload a self-contained predictor previously written by ``save``.

        Args:
            path: Path to the pickled predictor.

        Returns:
            The reconstructed ``FlaxPredictor``.
        """
        # The file is one this package wrote (save), in the job's own workspace.
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        return cls(
            payload["params_list"],
            payload["transform"],
            payload["architecture"],
            payload["prediction_length"],
            payload["context_length"],
            locations=payload.get("locations"),
            covariates=payload.get("covariates", REQUIRED_COVARIATES),
        )


class DeepAutoRegressiveModel:
    """Deep auto-regressive forecaster for disease case counts.

    The estimator wraps the [`ARModel2`][chapkit_deep_auto_regressive.core.rnn_model.ARModel2] network: it
    extracts and scales features, slices each location's series into
    ``context_length + prediction_length`` windows, and fits the network by
    maximizing the negative-binomial likelihood of the observed cases. Calling
    ``train`` returns a [`FlaxPredictor`][chapkit_deep_auto_regressive.core.model.FlaxPredictor].

    All input/output is via tidy :class:`pandas.DataFrame` objects with the columns
    ``location``, ``time_period``, ``rainfall``, ``mean_temperature``,
    ``population`` and (for training) ``disease_cases``. Listing extra column
    names in ``additional_covariates`` feeds them to the network as further
    features, on top of the always-present required three.

    The defaults are the tuned configuration: a GRU encoder/decoder, a 3-year
    context, and a 5-member deep ensemble.

    Attributes:
        prediction_length: Number of periods to forecast ahead.
        n_iter: Number of training epochs (per ensemble member).
        context_length: Number of past periods read as context.
        learning_rate: Adam learning rate; with a schedule, its start and end rate.
        lr_schedule: ``"constant"`` (Adam at ``learning_rate``) or
            ``"warmup_cosine"`` (warm up linearly to ``peak_learning_rate`` over
            ``warmup_epochs``, then cosine-decay back to ``learning_rate`` at the
            end of ``n_iter``).
        peak_learning_rate: Peak rate of the ``warmup_cosine`` schedule.
        warmup_epochs: Linear warmup epochs of the ``warmup_cosine`` schedule.
        n_ensemble: Independently seeded models to train and pool at predict time
            (a deep ensemble; improves predictive calibration / CRPS).
        additional_covariates: Extra covariate columns to use as features,
            appended after the required ``rainfall``, ``mean_temperature`` and
            ``population``. Empty by default.
        cell: Recurrent cell type (``"gru"`` or ``"simple"``).
        rnn_features: Hidden width of the encoder/decoder cells.
        preprocess_hidden / preprocess_output / embedding_dim / head_features:
            Network layer widths.
        distribution_head: Maps the network output to the sampling distribution.
    """

    # --- training ---
    prediction_length = 3
    n_iter: int = 400
    context_length = 36
    learning_rate = 1e-3
    #: ``"warmup_cosine"`` reaches the constant-rate score in far fewer epochs on
    #: large panels: on 1,500 simulated weekly facilities, peak 1e-3 over 150
    #: epochs matched constant 1e-5 over 1000.
    lr_schedule: str = "constant"
    peak_learning_rate: float = 1e-3
    warmup_epochs: int = 5
    n_ensemble: int = 5
    # Offsets every ensemble member's seed. The model is otherwise fully
    # deterministic, so without this two runs of the same configuration are
    # bit-identical and the seed variance -- the noise floor any change has to
    # clear -- cannot be measured at all.
    seed_offset: int = 0
    # --- early stopping ---
    # Held-out loss on these series bottoms out around epoch 30 and then climbs
    # for the remaining 370, so training to a fixed n_iter costs 1.3-2.5 nats.
    # Each member first fits on all but the held-out tail to find its own best
    # epoch, then refits on the full series for that many epochs -- so no data is
    # permanently given up, and the usual run is several times cheaper than the
    # fixed-length one it replaces.
    early_stopping: bool = True
    #: Extra periods the held-out block gets on top of the one window it must
    #: contain, so the block spans ``context_length + prediction_length +
    #: validation_periods`` periods and yields ``validation_periods + 1`` windows.
    validation_periods: int = 12
    #: Windows the probe must have to be able to locate a loss minimum. One
    #: window passes a length check but makes the probe meaningless, so this is a
    #: floor on usefulness rather than on arithmetic.
    min_probe_windows: int = 5
    #: How the probe's stopping point is carried over to the refit. ``"steps"``
    #: gives the refit the same number of gradient steps; ``"epochs"`` gives it
    #: the same number of passes over its own windows, which is more steps in
    #: proportion to the extra data it has. Neither is obviously right -- a larger
    #: training set supports more updates before it overfits, which argues for
    #: ``"epochs"``, but the ratio is unbounded, which argues for ``"steps"``.
    #: Chosen empirically; see the tests and the PR discussion.
    refit_scaling: str = "epochs"
    patience: int = 6
    eval_every: int = 5
    additional_covariates: Sequence[str] = ()
    # --- network architecture (persisted in the saved predictor) ---
    cell: str = "gru"
    rnn_features: int = 16
    preprocess_hidden: int = 16
    preprocess_output: int = 8
    embedding_dim: int = 8
    head_features: int = 24
    rnn_layers: int = 1
    recursive_decode: bool = False
    dropout_rate: float = 0.2
    input_dropout_rate: float = 0.0
    distribution_head = staticmethod(nb_head)

    def __init__(self, rng_key: jax.Array = jax.random.PRNGKey(100)) -> None:
        self.rng_key = rng_key
        self._model: nn.Module | None = None
        self._n_locations: int | None = None
        self._params: Any = None
        self._transform: Transform | None = None
        self._validation_loader: SimpleDataLoader | None = None
        self._locations: list[Any] | None = None
        self._validation_locations: list[Any] | None = None

    @property
    def covariates(self) -> tuple[str, ...]:
        """The feature covariates in order: the required three then any extras.

        Additional covariates that duplicate a required one are ignored, so the
        required covariates always lead exactly once.
        """
        extra = tuple(c for c in self.additional_covariates if c not in REQUIRED_COVARIATES)
        return REQUIRED_COVARIATES + extra

    @property
    def architecture(self) -> Architecture:
        """The network hyperparameters, persisted so predict rebuilds the same net."""
        return {
            "cell": self.cell,
            "rnn_features": self.rnn_features,
            "preprocess_hidden": self.preprocess_hidden,
            "preprocess_output": self.preprocess_output,
            "embedding_dim": self.embedding_dim,
            "head_features": self.head_features,
            "rnn_layers": self.rnn_layers,
            "recursive_decode": self.recursive_decode,
            "dropout_rate": self.dropout_rate,
            "input_dropout_rate": self.input_dropout_rate,
        }

    def set_model(self, model: nn.Module) -> None:
        """Override the network instance instead of building one lazily.

        Args:
            model: A flax module to use in place of the default architecture.
        """
        self._model = model

    @property
    def model(self) -> nn.Module:
        """The flax network, built lazily from the architecture on first access."""
        if self._model is None:
            # The location count only sizes the embedding at call time, from the
            # input shape; 1 stands in before training has seen the data.
            n_locations = self._n_locations if self._n_locations is not None else 1
            self._model = build_network(n_locations, **self.architecture)
        return self._model

    def learning_rate_for(self, steps_per_epoch: int) -> float | optax.Schedule:
        """Return the Adam learning rate for a loader: a constant or a per-step schedule.

        The schedule spans ``n_iter`` epochs of ``steps_per_epoch`` steps. Both
        an early-stopping probe and its refit get the schedule sized by the full
        series, so they follow the same schedule.

        Args:
            steps_per_epoch: Optimizer steps per epoch (one per training window).

        Raises:
            ValueError: If ``lr_schedule`` is not a known schedule.
        """
        if self.lr_schedule == "constant":
            return self.learning_rate
        if self.lr_schedule != "warmup_cosine":
            raise ValueError(f"lr_schedule must be 'constant' or 'warmup_cosine', got {self.lr_schedule!r}")
        total_steps = max(1, self.n_iter * steps_per_epoch)
        warmup_steps = min(self.warmup_epochs * steps_per_epoch, total_steps - 1)
        return optax.warmup_cosine_decay_schedule(
            init_value=self.learning_rate,
            peak_value=self.peak_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=self.learning_rate,
        )

    def _get_dataset(self, data: pd.DataFrame) -> DLDataSet:
        """Build a windowed dataset from the input frame."""
        x, y = get_series(data, self.covariates)
        return DLDataSet(x, y, forecast_length=self.prediction_length, context_length=self.context_length)

    def set_validation_data(self, historic: pd.DataFrame, future: pd.DataFrame) -> None:
        """Provide a held-out window for validation-loss reporting during training.

        The supplied history and future are concatenated into a single window and
        wrapped in a loader the trainer evaluates periodically. Unlike forecasting,
        the validation ``future`` must include observed ``disease_cases`` — they
        are the labels the validation loss is computed against.

        The fitted feature scaler is attached later, inside
        [`train`][chapkit_deep_auto_regressive.core.model.DeepAutoRegressiveModel.train], so the
        validation features are standardized the same way as the training ones.

        Args:
            historic: History including observed cases.
            future: The matching future periods, with both covariates and observed
                ``disease_cases``.

        Raises:
            ValueError: If ``future`` has no ``disease_cases`` column.
        """
        if "disease_cases" not in future.columns:
            raise ValueError(
                "validation future must include observed disease_cases (they are the "
                "labels the validation loss is computed against)"
            )
        if set(historic["location"].unique()) != set(future["location"].unique()):
            raise ValueError("validation historic and future must cover the same set of locations")
        self._validation_locations = sorted(historic["location"].unique())
        x, y = get_series(historic, self.covariates)
        x = x[:, -self.context_length :]
        y = y[:, -self.context_length :]
        fx, fy = get_series(future, self.covariates)
        full_x = np.concatenate([x, fx], axis=1)
        full_y = np.concatenate([y, fy], axis=1)
        self._validation_loader = SimpleDataLoader(
            DLDataSet(full_x, full_y, forecast_length=self.prediction_length, context_length=self.context_length)
        )

    def _early_stopping_loaders(self, data: pd.DataFrame) -> tuple[SimpleDataLoader, SimpleDataLoader] | None:
        """Split the tail off the training series to select a stopping epoch.

        The split is contiguous and final in time -- a random split would let the
        auto-regressive input see the future -- and the feature scaler is fitted on
        the earlier part only. Returns ``None`` when early stopping is switched
        off or when the caller supplied its own validation window.

        A window spans ``context_length + prediction_length`` periods. The
        held-out block needs one plus ``validation_periods``, and the fit side
        needs at least ``min_probe_windows`` of them, so the series must hold
        ``2 * (context_length + prediction_length) + validation_periods +
        min_probe_windows - 1`` periods -- 94 at the default 36 + 3 + 12 + 5.

        Too short a series raises rather than falling back to fixed-length
        training: the fallback is the overtraining this feature exists to
        prevent, and it would happen silently, since chap runs training in a
        subprocess whose logs the caller never sees.

        The probe-window floor matters as much as the total length. One window
        clears a naive length check but cannot locate a loss minimum -- the probe
        would evaluate ``patience`` times against a single fixed slice of history
        -- so lowering ``validation_periods`` until the error goes away would buy
        a fit that is worse than not using early stopping at all.

        Args:
            data: The training frame.

        Raises:
            ValueError: If the series is too short to hold out a validation window.

        Returns:
            A ``(fit_loader, validation_loader)`` pair, or ``None``.
        """
        if not self.early_stopping or self._validation_loader is not None:
            return None
        x, y = get_series(data, self.covariates)
        total_length = self.context_length + self.prediction_length
        # validation_periods is the *extra* history the held-out block gets beyond
        # the single window it must contain, so the knob has an effect at every
        # setting rather than being swallowed by a max() against the window size.
        n_validation = total_length + max(self.validation_periods, 0)
        split = x.shape[1] - n_validation
        probe_windows = split - total_length + 1
        if probe_windows < self.min_probe_windows:
            needed = n_validation + total_length + self.min_probe_windows - 1
            raise ValueError(
                f"early stopping needs at least {needed} periods "
                f"({n_validation} held out, and enough left to give the probe "
                f"{self.min_probe_windows} windows), but the shortest series has {x.shape[1]} "
                f"and would give the probe {max(probe_windows, 0)}. Either shorten "
                f"context_length (currently {self.context_length}), lower validation_periods "
                f"(currently {self.validation_periods}), or set early_stopping=False to train "
                f"for a fixed n_iter={self.n_iter}."
            )
        fit_set = DLDataSet(
            x[:, :split], y[:, :split], forecast_length=self.prediction_length, context_length=self.context_length
        )
        scaler = ZScaler.from_data(fit_set)
        fit_set.set_transform(scaler)
        validation_set = DLDataSet(
            x[:, split:], y[:, split:], forecast_length=self.prediction_length, context_length=self.context_length
        )
        validation_set.set_transform(scaler)
        return SimpleDataLoader(fit_set), SimpleDataLoader(validation_set)

    def train(self, data: pd.DataFrame) -> FlaxPredictor:
        """Fit the model and return a predictor.

        Extracts features, fits the feature scaler, builds the windowed loader,
        and runs the [`Trainer`][chapkit_deep_auto_regressive.core.trainer.Trainer] to minimize the negative
        log-likelihood.

        Args:
            data: Training frame, one row per location and period, with observed
                ``disease_cases``.

        Returns:
            A [`FlaxPredictor`][chapkit_deep_auto_regressive.core.model.FlaxPredictor] holding the trained
            parameters and the fitted scaler.
        """
        self._locations = sorted(data["location"].unique())
        self._n_locations = int(data["location"].nunique())
        data_set = self._get_dataset(data)
        transform = ZScaler.from_data(data_set)
        self._transform = transform
        data_set.set_transform(transform)
        early_stopping_loaders = self._early_stopping_loaders(data)
        # Standardize the validation window with the same fitted scaler, otherwise
        # its features stay on the raw scale and the reported validation loss is
        # not comparable to the training loss.
        if self._validation_loader is not None:
            if self._validation_locations != self._locations:
                raise ValueError(
                    "validation locations must match the training locations "
                    f"{self._locations}, but got {self._validation_locations}"
                )
            self._validation_loader.dataset.set_transform(transform)
        data_loader = SimpleDataLoader(data_set)
        params_list: list[Any] = []
        for member in range(max(1, self.n_ensemble)):
            seed = self.seed_offset + member
            n_iter = self.n_iter
            max_steps = None
            if early_stopping_loaders is not None:
                fit_loader, validation_loader = early_stopping_loaders
                probe = Trainer(
                    self.model,
                    self.n_iter,
                    # The probe's schedule is sized by the full series, so the
                    # probe and the refit follow the same schedule.
                    learning_rate=self.learning_rate_for(len(data_set)),
                    validation_loader=validation_loader,
                    seed=seed,
                    eval_every=self.eval_every,
                    patience=self.patience,
                )
                probe.train(fit_loader, self._loss)
                # Carry the budget across as gradient *steps*, not epochs. The
                # probe runs over the pre-split prefix and the refit over the
                # whole frame, so an epoch is a different amount of training in
                # each -- by 4x at the shortest supported series, and far more
                # once context_length is tuned down. Passing the epoch index
                # would silently reinstate the overtraining this prevents.
                max_steps = max(probe.best_step or 0, 1)
                if self.refit_scaling == "epochs":
                    # Same number of passes over the data, which is more gradient
                    # steps in proportion to the refit's extra windows.
                    ratio = len(data_set) / max(len(fit_loader.dataset), 1)
                    max_steps = max(int(round(max_steps * ratio)), 1)
                elif self.refit_scaling != "steps":
                    raise ValueError(f"refit_scaling must be 'steps' or 'epochs', got {self.refit_scaling!r}")
                logger.info(
                    "member %d: best validation loss %.4f at epoch %d; refitting on the full "
                    "series for %d steps (%s scaling)",
                    member,
                    probe.best_validation_loss,
                    probe.best_epoch,
                    max_steps,
                    self.refit_scaling,
                )
            trainer = Trainer(
                self.model,
                n_iter,
                learning_rate=self.learning_rate_for(len(data_set)),
                validation_loader=None if early_stopping_loaders is not None else self._validation_loader,
                seed=seed,
                eval_every=self.eval_every,
                patience=self.patience,
                max_steps=max_steps,
            )
            params_list.append(trainer.train(data_loader, self._loss).params)
        self._params = params_list[0]
        return FlaxPredictor(
            params_list,
            transform,
            self.architecture,
            self.prediction_length,
            self.context_length,
            locations=self._locations,
            covariates=self.covariates,
        )

    def load_predictor(self, path: str) -> FlaxPredictor:
        """Load a saved predictor. Self-contained — rebuilt entirely from the file.

        Args:
            path: Path to a file written by
                [`FlaxPredictor.save`][chapkit_deep_auto_regressive.core.model.FlaxPredictor.save].

        Returns:
            The reconstructed [`FlaxPredictor`][chapkit_deep_auto_regressive.core.model.FlaxPredictor].
        """
        return FlaxPredictor.load(path)

    def predict(self, historic: pd.DataFrame, future: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
        """Forecast the future periods directly from this (trained) model.

        Equivalent to the predictor's ``predict``; useful when forecasting from a
        model trained in the same process.

        Args:
            historic: Recent history including observed ``disease_cases``.
            future: The periods to forecast, with covariates but no cases.
            num_samples: Number of samples per location and period.

        Returns:
            A frame with columns ``time_period``, ``location`` and one ``sample_i``
            column per draw.
        """
        _check_predict_inputs(historic, future, self.prediction_length, self.context_length, self._locations)
        x, _ = get_series(future, self.covariates)
        prev_values, prev_y = get_series(historic, self.covariates)
        prev_values = prev_values[:, -self.context_length :]
        prev_y = prev_y[:, -self.context_length :]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        if self._transform is None:
            raise ValueError("the model has not been trained; call train first")
        dataset.set_transform(self._transform)
        window_x, window_y = dataset.prediction_instance()
        eta = _apply_network(self.model, self._params, window_x, window_y)
        n_prev = prev_values.shape[1]
        samples = self.get_samples(eta[:, n_prev - 1 :], num_samples)
        return _forecast_frame(future, samples)

    def loss_func(self, eta_pred: Any, y_true: Any) -> jnp.ndarray:
        """Per-period negative log-likelihood of the observed cases.

        Args:
            eta_pred: The network output, shape ``(..., periods, 2)``.
            y_true: The window's target; ``y_true[..., 0]`` is the lagged value, so
                the likelihood is evaluated against ``y_true[..., 1:]``.

        Returns:
            The element-wise negative log-likelihood under the negative binomial.
        """
        return -self.distribution_head(eta_pred).log_prob(y_true[..., 1:])

    def _loss(self, y_pred: Any, y_true: Any) -> Any:
        """Aggregate the per-period loss over the whole window, emphasizing the horizon.

        Training on every period (not just the forecast horizon) gives the
        auto-regressive encoder one-step-ahead supervision across the whole
        context, which is far more signal than the 3 horizon periods alone; the
        extra horizon term keeps the forecast region weighted up.
        """
        L = self.loss_func(y_pred, y_true)
        return jnp.mean(L) + jnp.mean(L[:, -self.prediction_length :])

    def get_samples(self, eta: jax.Array, n_samples: int) -> NDArray[Any]:
        """Draw samples from the output distribution for each period.

        Args:
            eta: The network's per-period output, shape ``(..., periods, 2)``.
            n_samples: Number of samples to draw per period.

        Returns:
            Sampled counts, with a leading sample axis of length ``n_samples``.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))
