"""The network and the trainer on toy windows."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chapkit_deep_auto_regressive.core.data_loader import DataSet, SimpleDataLoader
from chapkit_deep_auto_regressive.core.options import ModelOptions
from chapkit_deep_auto_regressive.core.rnn_model import build_network
from chapkit_deep_auto_regressive.core.trainer import Trainer


def _param_count(params: Any) -> int:
    leaves: list[jax.Array] = jax.tree_util.tree_leaves(params)
    return sum(leaf.size for leaf in leaves)


def _toy_loader(n_loc: int = 2, n_periods: int = 12, n_feat: int = 4, seed: int = 0) -> SimpleDataLoader:
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n_loc, n_periods, n_feat)).astype("float32")
    y = rng.poisson(3, (n_loc, n_periods)).astype("float32")
    return SimpleDataLoader(DataSet(X, y, forecast_length=2, context_length=4))


def test_network_forward_produces_two_finite_channels() -> None:
    loader = _toy_loader()
    x, ar_y, _ = next(iter(loader))
    model = build_network(x.shape[0])
    params = model.init(jax.random.PRNGKey(0), x, ar_y, training=False)
    eta = np.asarray(model.apply(params, x, ar_y))
    assert eta.shape[-1] == 2
    assert np.isfinite(eta).all()


@pytest.mark.parametrize("recursive_decode", [True, False])
def test_rnn_layers_adds_parameters(recursive_decode: bool) -> None:
    """rnn_layers must stack real layers in both decode paths.

    Regression test: the recursive_decode branch previously ignored n_layers, so
    rnn_layers=2 trained a network identical to rnn_layers=1.
    """
    loader = _toy_loader()
    x, ar_y, _ = next(iter(loader))
    key = jax.random.PRNGKey(0)

    def count(n_layers: int) -> int:
        architecture = ModelOptions(rnn_layers=n_layers, recursive_decode=recursive_decode).architecture
        net = build_network(x.shape[0], architecture)
        return _param_count(net.init(key, x, ar_y, training=False))

    one, two, three = count(1), count(2), count(3)
    assert one < two < three, f"rnn_layers had no effect: {(one, two, three)}"


def test_trainer_runs_a_couple_of_steps() -> None:
    loader = _toy_loader()
    x, _, _ = next(iter(loader))
    model = build_network(x.shape[0])
    state = Trainer(model, n_iter=2, learning_rate=1e-3).train(loader, lambda eta, y: jnp.mean(eta**2))
    assert state.params is not None
