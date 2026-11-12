import jax
import jax.numpy as jnp
import numpy as np

from chapkit_deep_auto_regressive.core.distributions import NegativeBinomial3, nb_head


def test_nb3_sampling_is_reproducible_from_key() -> None:
    # The JAX key must control the draw: same key -> same samples, different key
    # -> different samples (previously the key was ignored and scipy used the
    # global RNG, making forecasts non-reproducible).
    dist = NegativeBinomial3(jnp.array([5.0, 5.0, 5.0]), jnp.array([0.1, 0.1, 0.1]))
    a = np.asarray(dist.sample(jax.random.PRNGKey(0), (50,)))
    b = np.asarray(dist.sample(jax.random.PRNGKey(0), (50,)))
    c = np.asarray(dist.sample(jax.random.PRNGKey(1), (50,)))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_negative_binomial3_direct_instantiation() -> None:
    # Regression: NB3 used to be a frozen dataclass with a hand-written __init__
    # that only worked through subclassing. Direct construction must work.
    dist = NegativeBinomial3(jnp.array([3.0]), jnp.array([0.1]))
    assert np.isfinite(np.asarray(dist.log_prob(jnp.array([2.0])))).all()


def test_nb_head_log_prob_shape_and_finite() -> None:
    eta = jnp.array([[0.5, -0.3], [1.0, 0.2]])
    dist = nb_head(eta)
    lp = np.asarray(dist.log_prob(jnp.array([1.0, 2.0])))
    assert lp.shape == (2,)
    assert np.isfinite(lp).all()


def test_nb_head_skips_nan() -> None:
    eta = jnp.array([[0.5, -0.3], [1.0, 0.2]])
    lp = np.asarray(nb_head(eta).log_prob(jnp.array([np.nan, 2.0])))
    assert lp[0] == 0.0  # a missing observation contributes nothing
    assert np.isfinite(lp[1])
