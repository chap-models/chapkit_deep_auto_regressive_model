"""Probability distributions for the network's output head.

The model uses [`nb_head`][chapkit_deep_auto_regressive.core.distributions.nb_head], a NaN-tolerant
negative binomial. The other distributions are small, self-contained primitives.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax.scipy import stats
from jax.typing import ArrayLike
from numpy.typing import NDArray


@dataclass(frozen=True)
class Normal:
    """Normal distribution parameterised by mean and standard deviation."""

    mu: Any
    sigma: Any
    ndim: int = 0

    def sample(self, key: Any, shape: Sequence[int] = ()) -> Any:
        """Sample from the normal distribution."""
        if tuple(shape) != ():
            raise ValueError(f"Normal.sample draws one value per parameter; got shape {shape!r}")
        if hasattr(self.mu, "shape"):
            shape = self.mu.shape
        return jax.random.normal(key, shape) * self.sigma + self.mu

    def log_prob(self, x: Any) -> Any:
        """Log probability density at ``x``."""
        if self.ndim > 0:
            if self.ndim != 1:
                raise ValueError(f"Normal supports ndim 0 or 1, got {self.ndim}")
            return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma).sum()
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)


@dataclass(frozen=True)
class NegativeBinomial2:
    """Negative binomial in mean/dispersion (``mu``, ``alpha``) parameterisation."""

    mu: Any
    alpha: Any

    def log_prob(self, x: Any) -> Any:
        """Log probability mass at ``x``."""
        log_gamma = (
            jax.scipy.special.gammaln(x + 1 / self.alpha)
            - jax.scipy.special.gammaln(1 / self.alpha)
            - jax.scipy.special.gammaln(x + 1)
        )
        first_term = 1 / self.alpha * jnp.log(1 + self.alpha * self.mu)
        second_term = x * (jnp.log(self.mu) + jnp.log(self.alpha) - jnp.log(1 + self.alpha * self.mu))
        return log_gamma - first_term + second_term

    def mean(self) -> Any:
        """Distribution mean."""
        return self.mu

    def sigma(self) -> Any:
        """Distribution variance."""
        return self.mu + self.mu**2 * self.alpha

    def p(self) -> Any:
        """Success probability in the standard parameterisation."""
        return self.alpha / (1 + self.alpha * self.mu)

    def n(self) -> Any:
        """Number of failures in the standard parameterisation."""
        return 1 / self.alpha


class NegativeBinomial3:
    """Negative binomial parameterised by total count and logits."""

    def __init__(self, total_count: jax.Array, logits: jax.Array) -> None:
        """Store count/logits and derive the success probabilities."""
        self.total_count = total_count
        self.logits = logits
        self.probs = jax.nn.sigmoid(logits)

    @property
    def mean(self) -> Any:
        """Distribution mean."""
        return self.total_count * jnp.exp(self.logits)

    @property
    def variance(self) -> Any:
        """Distribution variance."""
        return self.mean / jax.nn.sigmoid(-self.logits)

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Log probability mass at ``value``."""
        log_unnormalized_prob = self.total_count * jax.nn.log_sigmoid(-self.logits) + value * jax.nn.log_sigmoid(
            self.logits
        )
        log_normalization = (
            -jax.lax.lgamma(self.total_count + value) + jax.lax.lgamma(1.0 + value) + jax.lax.lgamma(self.total_count)
        )
        return log_unnormalized_prob - log_normalization

    def sample(self, key: jax.Array, shape: int | tuple[int, ...] = ()) -> NDArray[Any]:
        """Sample counts via the scipy negative binomial, seeded from ``key``.

        The JAX ``key`` is folded into a seed for scipy's generator so that draws
        are reproducible and controlled by the model's RNG, rather than depending
        on NumPy's global random state.
        """
        if isinstance(shape, int):
            shape = (shape,)
        seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        samples = self.scipy_nbinom.rvs(shape + self.total_count.shape, random_state=seed)
        return np.moveaxis(samples, 0, -1)

    @property
    def scipy_nbinom(self) -> Any:
        """Equivalent scipy negative binomial frozen distribution."""
        return scipy.stats.nbinom(n=self.total_count, p=1.0 - self.probs)


@dataclass(frozen=True)
class Poisson:
    """Poisson distribution parameterised by rate."""

    rate: Any

    def sample(self, key: Any, shape: tuple[int, ...] = ()) -> Any:
        """Sample counts from the Poisson distribution."""
        if shape != ():
            raise ValueError(f"Poisson.sample draws one value per rate; got shape {shape!r}")
        if hasattr(self.rate, "shape"):
            shape = self.rate.shape
        return jax.random.poisson(key, self.rate, shape)

    def log_prob(self, x: Any) -> Any:
        """Log probability mass at ``x``."""
        return stats.poisson.logpmf(x, self.rate)


class NBSkipNaN(NegativeBinomial3):
    """Negative binomial whose NaN observations contribute zero log-probability.

    Gaps in surveillance leave NaN targets; they must not add to the loss.
    """

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Log probability mass at ``value``, and 0 where ``value`` is NaN."""
        nans = jnp.isnan(value)
        masked = jnp.where(nans, 0, value)
        return jnp.where(nans, 0, super().log_prob(masked))


def nb_head(eta: jax.Array) -> NBSkipNaN:
    """Interpret a model's two-channel output ``eta`` as a negative-binomial.

    Channel 0 (passed through softplus) is the count and channel 1 is the logits.
    """
    return NBSkipNaN(jax.nn.softplus(eta[..., 0]), eta[..., 1])
