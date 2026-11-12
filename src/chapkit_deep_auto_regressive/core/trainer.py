"""The training loop.

[`Trainer`][chapkit_deep_auto_regressive.core.trainer.Trainer] fits a flax network by minimizing a
caller-supplied loss with the Adam optimizer. The per-step update is JIT-compiled
with ``jax.jit`` and uses ``jax.value_and_grad`` for the gradient; a small L2
penalty on the weight matrices is added for regularization.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.typing import ArrayLike

from .data_loader import WindowLoader

#: Maps the network output ``eta`` and the window's target to a scalar loss.
LossFn = Callable[[jax.Array, ArrayLike], jax.Array]

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """Flax training state extended with a PRNG key for dropout.

    Attributes:
        key: The PRNG key folded per step to drive dropout.
    """

    key: jax.Array


#: One compiled gradient step: ``(state, dropout_key, x, ar_y, y) -> (state, loss)``.
TrainStep = Callable[[TrainState, jax.Array, ArrayLike, ArrayLike, ArrayLike], tuple[TrainState, jax.Array]]
#: The compiled validation loss: ``(state, x, ar_y, y) -> loss``.
ValidationLoss = Callable[[TrainState, ArrayLike, ArrayLike, ArrayLike], jax.Array]


def _apply(state: "TrainState", params: Any, *inputs: ArrayLike, **kwargs: Any) -> jax.Array:
    """Run the network through ``state.apply_fn``.

    flax types ``TrainState.apply_fn`` as a bare ``Callable``, so the network
    output is cast to the array it is.
    """
    return cast(jax.Array, state.apply_fn(params, *inputs, **kwargs))


def l2_regularization(params: Any, scale: float = 1.0) -> jax.Array:
    """Sum of squared weight-matrix entries, used as an L2 penalty.

    Only rank-2 parameters (weight matrices) are penalized; biases and other
    parameters are left out.

    Args:
        params: A pytree of model parameters.
        scale: Multiplier applied to the summed squared weights.

    Returns:
        The scaled L2 penalty as a scalar.
    """
    leaves: list[jax.Array] = jax.tree_util.tree_leaves(params)
    return sum((jnp.sum(jnp.square(p)) for p in leaves if p.ndim == 2), start=jnp.zeros(())) * scale


class Trainer:
    """Fits a flax model to windowed data with Adam.

    The trainer initializes the model from the first window, then for ``n_iter``
    epochs runs a JIT-compiled gradient step over every training window. When a
    validation loader is supplied, the validation loss is reported every ten
    epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        n_iter: int = 3000,
        learning_rate: float | optax.Schedule = 1e-5,
        validation_loader: WindowLoader | None = None,
        seed: int = 0,
        eval_every: int = 5,
        patience: int = 6,
        max_steps: int | None = None,
    ) -> None:
        """Configure the trainer.

        Args:
            model: The flax module to fit.
            n_iter: Number of epochs (full passes over the windows).
            learning_rate: Adam learning rate, or an optax schedule over steps.
            validation_loader: Optional loader providing a held-out window for
                periodic validation-loss reporting.
            seed: Seed for parameter initialization and dropout, so independently
                seeded trainers yield distinct ensemble members.
            eval_every: Epochs between validation evaluations.
            patience: Stop after this many consecutive evaluations without a new
                best validation loss. Only active when a validation loader is
                supplied; ``0`` disables stopping and restores the previous
                fixed-length behaviour.
            max_steps: Stop after this many gradient steps, regardless of epoch
                boundaries. A step budget is the portable unit between two
                loaders: an epoch is a pass over whichever windows the loader
                holds, so the same epoch count means different amounts of
                training on datasets of different length.
        """
        self.model = model
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._validation_loader = validation_loader
        self.seed = seed
        self.eval_every = eval_every
        self.patience = patience
        self.max_steps = max_steps
        #: Epoch at which the best validation loss was seen, set by ``train``.
        self.best_epoch: int | None = None
        #: Gradient steps taken to reach the best validation loss, set by
        #: ``train``. This, not ``best_epoch``, is what transfers to a run over a
        #: different set of windows.
        self.best_step: int | None = None
        #: The best validation loss seen, set by ``train``.
        self.best_validation_loss: float | None = None

    def train(self, data_loader: WindowLoader, loss_fn: LossFn) -> TrainState:
        """Train the model and return the final state.

        Initializes parameters from the first window, then repeatedly applies a
        JIT-compiled Adam step over all windows. Each step adds an L2 penalty and
        folds a fresh dropout key from the step counter for stochastic
        regularization.

        Args:
            data_loader: Yields ``(x, ar_y, y)`` training windows.
            loss_fn: Maps ``(eta, y)`` to a scalar loss (the negative
                log-likelihood under the model's output distribution).

        Returns:
            The final [`TrainState`][chapkit_deep_auto_regressive.core.trainer.TrainState] holding the
            trained parameters.
        """
        ix, iar_y, _ = next(iter(data_loader))
        params = self.model.init(jax.random.PRNGKey(self.seed), ix, iar_y, training=False)
        dropout_key = jax.random.PRNGKey(self.seed + 40)

        training_state: TrainState = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optax.adam(self.learning_rate), key=dropout_key
        )

        def train_step_fn(
            state: TrainState, dropout_key: jax.Array, x: ArrayLike, ar_y: ArrayLike, y: ArrayLike
        ) -> tuple[TrainState, jax.Array]:
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_func(params: Any) -> jax.Array:
                eta = _apply(state, params, x, ar_y, training=True, rngs={"dropout": dropout_train_key})
                return loss_fn(eta, y) + l2_regularization(params, 0.001)

            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        def validation_loss_fn(state: TrainState, x: ArrayLike, ar_y: ArrayLike, y: ArrayLike) -> jax.Array:
            return loss_fn(_apply(state, state.params, x, ar_y, training=False), y)

        # jax.jit's wrapper is untyped; the compiled functions keep the signatures above.
        train_step = cast(TrainStep, jax.jit(train_step_fn))
        get_validation_loss = cast(ValidationLoss, jax.jit(validation_loss_fn))

        def validation_loss(loader: WindowLoader) -> float:
            total = 0.0
            for v_x, v_ar, v_y in loader:
                total += float(get_validation_loss(training_state, v_x, v_ar, v_y))
            return total

        validation_loader = self._validation_loader
        cur_loss: jax.Array | None = None
        best_state: TrainState = training_state
        best_loss, best_epoch, since_best = float("inf"), 0, 0
        for i in range(self.n_iter):
            for x, ar_y, y in data_loader:
                if self.max_steps is not None and int(training_state.step) >= self.max_steps:
                    break
                training_state, cur_loss = train_step(training_state, dropout_key, x, ar_y, y)
            if self.max_steps is not None and int(training_state.step) >= self.max_steps:
                logger.info("reached the %d-step budget at epoch %d", self.max_steps, i)
                break
            if validation_loader is None:
                if i % 10 == 0:
                    logger.info("epoch %d: loss=%s", i, cur_loss)
                continue
            if i % self.eval_every:
                continue
            current = validation_loss(validation_loader)
            logger.info("epoch %d: loss=%s validation_loss=%s", i, cur_loss, current)
            if current < best_loss:
                # Keeping the best parameters, not the last, is the whole point:
                # held-out loss on these series bottoms out early and then climbs
                # for the rest of the run.
                best_state = training_state
                best_loss, best_epoch, since_best = current, i, 0
                continue
            since_best += 1
            if self.patience and since_best >= self.patience:
                logger.info("early stopping at epoch %d; best was epoch %d (%.4f)", i, best_epoch, best_loss)
                break

        if validation_loader is None:
            return training_state
        self.best_epoch, self.best_validation_loss = best_epoch, best_loss
        self.best_step = int(best_state.step)
        return best_state
