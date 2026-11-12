"""The recurrent network architecture.

The model that the auto-regressive estimator trains is
[`ARModel2`][chapkit_deep_auto_regressive.core.rnn_model.ARModel2], assembled by the
[`model_makers`][chapkit_deep_auto_regressive.core.rnn_model.model_makers] factories. It processes a window
in four steps: a per-location [`Preprocess`][chapkit_deep_auto_regressive.core.rnn_model.Preprocess] stage,
an auto-regressive join ([`ARAdder`][chapkit_deep_auto_regressive.core.rnn_model.ARAdder]), and two stacked
``SimpleCell`` RNNs that encode the context and decode the forecast horizon.

Array convention: ``batch x location x time x feature``.
"""

from collections.abc import Callable
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import SimpleCell

# Dimensions: batch_dim x location_dim x time_dim x feature_dim

#: A recurrent cell the network can use.
Cell = SimpleCell | nn.GRUCell

#: Recurrent cell types selectable by name. ``"gru"`` (gated) handles the
#: multi-step forecast horizon markedly better than the plain ``"simple"`` RNN.
CELL_TYPES: dict[str, type[Cell]] = {"simple": SimpleCell, "gru": nn.GRUCell}


def _run_rnn(cell: Cell, inputs: jax.Array, initial_carry: Any = None) -> jax.Array:
    """Run ``cell`` over the time axis and return its outputs.

    Called inside a module's compact ``__call__``, so the ``nn.RNN`` it creates is
    a submodule of that module, named as if it were created inline.
    """
    outputs: jax.Array | tuple[Any, jax.Array] = nn.RNN(cell)(inputs, initial_carry=initial_carry)
    if isinstance(outputs, tuple):  # only with return_carry=True, which is not used here
        raise TypeError("nn.RNN returned its carry; expected the outputs only")
    return outputs


def _dropout(x: jax.Array, rate: float, *, deterministic: bool, broadcast_dims: tuple[int, ...] = ()) -> jax.Array:
    """Apply ``nn.Dropout`` (created inline, as a submodule of the calling module).

    flax leaves ``Dropout.__call__`` unannotated, so its result is cast to the
    array it is.
    """
    return cast(jax.Array, nn.Dropout(rate=rate, broadcast_dims=broadcast_dims, deterministic=deterministic)(x))


def _cell_step(cell: Cell, carry: Any, x: jax.Array) -> tuple[Any, jax.Array]:
    """Advance a recurrent cell one step; return ``(new_carry, output)``.

    flax leaves the cells' ``__call__`` unannotated, so the result is cast to
    the ``(carry, output)`` pair it is.
    """
    return cast(tuple[Any, jax.Array], cell(carry, x))


class Preprocess(nn.Module):
    """Per-location feature embedding and projection.

    Each location is given a learned embedding vector (so the shared network can
    still distinguish regions), which is concatenated onto the per-period features
    and passed through a small dense stack with dropout. The result is a
    low-dimensional per-period representation the RNNs consume.

    Attributes:
        n_hidden: Width of the hidden dense layer.
        n_locations: Number of locations (unused at call time; the count is taken
            from the input shape).
        embedding_dim: Size of the per-location embedding.
        output_dim: Size of the produced per-period representation.
        dropout_rate: Dropout probability applied during training.
    """

    n_hidden: int = 4
    n_locations: int = 1
    embedding_dim: int = 4
    output_dim: int = 1
    dropout_rate: float = 0.2
    input_dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """Embed locations, concatenate, and project the features.

        Args:
            x: Features, shape ``(..., locations, time, features)``.
            training: Whether dropout is active.

        Returns:
            The projected features, shape ``(..., locations, time, output_dim)``.
        """
        if self.input_dropout_rate > 0.0:
            # Feature dropout: zero entire covariate channels (one mask per feature,
            # shared across location/time) so the network can't over-rely on any
            # single (possibly noisy) covariate — robustness with many covariates.
            x = _dropout(
                x,
                self.input_dropout_rate,
                broadcast_dims=tuple(range(x.ndim - 1)),
                deterministic=not training,
            )
        n_locations = x.shape[-3]
        loc = nn.Embed(num_embeddings=n_locations, features=self.embedding_dim)(jnp.arange(n_locations))
        axis = -2
        loc = jnp.repeat(loc[..., None, :], x.shape[axis], axis=axis)
        if x.ndim == 4:
            loc = jnp.repeat(loc[None, ...], x.shape[0], axis=0)
        x = jnp.concatenate([x, loc], axis=-1)  # batch x embedding_dim
        layers = [self.n_hidden]
        for i in range(len(layers)):
            x = nn.Dense(features=layers[i])(x)
            x = nn.relu(x)
        x = _dropout(x, self.dropout_rate, deterministic=not training)
        x = nn.Dense(features=self.output_dim)(x)
        return _dropout(x, self.dropout_rate, deterministic=not training)


class ARAdder(nn.Module):
    """The auto-regressive join: append past observations to the features.

    This is the step that makes the model auto-regressive. It concatenates the
    observed target ``y`` onto the processed features so the recurrent encoder
    sees the series' own recent values alongside the covariates.
    """

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Concatenate past targets onto the features.

        Args:
            x: Processed features, shape ``(..., time, features)``.
            y: Observed targets over the context, shape ``(..., n_context)``.

        Returns:
            Features with the lagged target appended on the feature axis.
        """
        n_y = y.shape[-1]
        # log1p the auto-regressive input so it sits on a comparable scale to the
        # z-scored covariate features instead of swamping them with raw counts.
        return jnp.concatenate([jnp.log1p(y)[..., None], x[..., 1 : n_y + 1, :]], axis=-1)


class MultiValueARAdder(nn.Module):
    """Auto-regressive join that also mixes targets across locations.

    A variant of [`ARAdder`][chapkit_deep_auto_regressive.core.rnn_model.ARAdder] that, in addition to each
    series' own past, passes the targets through a dense layer across locations so
    a region can borrow signal from the others.
    """

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Concatenate the lagged target and a cross-location mix onto the features.

        Args:
            x: Processed features, shape ``(..., time, features)``.
            y: Observed targets over the context, shape ``(locations, n_context)``.

        Returns:
            Features with both the per-location lag and a learned cross-location
            target mix appended on the feature axis.
        """
        n_y = y.shape[-1]
        collected_y = jnp.moveaxis(nn.Dense(features=y.shape[0])(jnp.moveaxis(y, 0, -1)), -1, 0)
        collected_y = nn.relu(collected_y)
        return jnp.concatenate([collected_y[..., None], y[..., None], x[..., 1 : n_y + 1, :]], axis=-1)


class ARModel2(nn.Module):
    """The full encoder/decoder auto-regressive network.

    The forward pass preprocesses the features, joins in the past targets, encodes
    the context window with one ``SimpleCell`` RNN, then continues from that RNN's
    final state across the forecast horizon with a second ``SimpleCell`` RNN
    (where observed targets are no longer available). Dense layers map the
    combined hidden states to the two output channels (``eta``) consumed by the
    negative-binomial head.

    Attributes:
        preprocess: The per-location feature stage.
        cell_pre: The recurrent cell used to encode the context.
        cell_post: The recurrent cell used to decode the forecast horizon.
        ar_adder: The auto-regressive join module.
        output_dim: Number of output channels per period (2 for the NB head).
        head_features: Width of the hidden dense layer mapping the combined
            encoder/decoder states to ``eta``.
    """

    preprocess: nn.Module
    cell_pre: Cell
    cell_post: Cell
    ar_adder: ARAdder | MultiValueARAdder = ARAdder()
    output_dim: int = 2
    head_features: int = 6
    n_layers: int = 1
    recursive_decode: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array, training: bool = False) -> jax.Array:
        """Run the encoder/decoder forward pass over a window.

        Args:
            x: Features over the full window, shape ``(..., time, features)``.
            y: Observed targets over the context, shape ``(..., n_context)``.
            training: Whether dropout in the preprocess stage is active.

        Returns:
            The per-period output ``eta``, shape ``(..., time, output_dim)``.
        """
        x = self.preprocess(x, training=training)
        n_y = y.shape[-1]
        prev_x = self.ar_adder(x, y)
        future_x = x[..., n_y + 1 :, :]
        if self.recursive_decode:
            # Decode the horizon one step at a time, feeding each step's predicted
            # (log) mean back as the next step's auto-regressive input — the same
            # recursion a classical AR model uses, which keeps the multi-step
            # forecast anchored instead of drifting. Train and predict run this
            # identically (free-running), so there is no train/predict mismatch.
            d1, d2 = nn.Dense(self.head_features), nn.Dense(self.output_dim)

            def head(h: jax.Array) -> jax.Array:
                return d2(nn.relu(d1(h)))

            prev_val = jnp.log1p(jnp.maximum(y[..., -1:], 0.0))  # last observed (log1p)
            if self.n_layers <= 1:
                # Single layer: encode the context with one RNN, decode the
                # horizon with one cell (unchanged from the original behaviour).
                states = _run_rnn(self.cell_pre, prev_x)
                ctx_eta = head(states)
                carries: list[Any] = [states[..., -1, :]]
                dec_cells: list[Cell] = [self.cell_post]
            else:
                # Stacked recursive decode: encode the context with n_layers
                # stacked RNNs (each layer's final state seeds the matching
                # decoder cell), then roll the horizon forward one step at a time
                # through the whole decoder stack. Mirrors the n_layers branch of
                # the non-recursive path, but keeps the step-by-step feedback loop.
                cell_cls, feats = type(self.cell_pre), self.cell_pre.features
                enc, carries = prev_x, []
                for _ in range(self.n_layers):
                    enc = _run_rnn(cell_cls(features=feats), enc)
                    carries.append(enc[..., -1, :])
                ctx_eta = head(enc)
                dec_cells = [cell_cls(features=feats) for _ in range(self.n_layers)]
            etas: list[jax.Array] = []
            for t in range(future_x.shape[-2]):
                # The bottom decoder cell sees the fed-back prediction; each higher
                # cell consumes the cell below it. Carries thread across horizon steps.
                layer_in = jnp.concatenate([prev_val, future_x[..., t, :]], axis=-1)
                for layer, cell in enumerate(dec_cells):
                    carries[layer], layer_in = _cell_step(cell, carries[layer], layer_in)
                eta_t = head(layer_in)
                etas.append(eta_t)
                mean = jax.nn.softplus(eta_t[..., 0]) * jnp.exp(eta_t[..., 1])  # NB mean
                prev_val = jnp.log1p(jnp.maximum(mean, 0.0))[..., None]
            return jnp.concatenate([ctx_eta, jnp.stack(etas, axis=-2)], axis=-2)
        if self.n_layers <= 1:
            states = _run_rnn(self.cell_pre, prev_x)
            new_states = _run_rnn(self.cell_post, future_x, initial_carry=states[..., -1, :])
        else:
            # Stacked encoder/decoder: each encoder layer's final state seeds the
            # matching decoder layer, giving the forecast horizon more depth.
            cell_cls, feats = type(self.cell_pre), self.cell_pre.features
            enc, encoder_carries = prev_x, list[jax.Array]()
            for _ in range(self.n_layers):
                enc = _run_rnn(cell_cls(features=feats), enc)
                encoder_carries.append(enc[..., -1, :])
            dec = future_x
            for layer in range(self.n_layers):
                dec = _run_rnn(cell_cls(features=feats), dec, initial_carry=encoder_carries[layer])
            states, new_states = enc, dec
        x = jnp.concatenate([states, new_states], axis=-2)
        x = nn.Dense(features=self.head_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


def build_network(
    n_locations: int,
    *,
    cell: str = "gru",
    rnn_features: int = 16,
    preprocess_hidden: int = 16,
    preprocess_output: int = 8,
    embedding_dim: int = 8,
    head_features: int = 24,
    rnn_layers: int = 1,
    recursive_decode: bool = False,
    dropout_rate: float = 0.2,
    input_dropout_rate: float = 0.0,
) -> ARModel2:
    """Build an [`ARModel2`][chapkit_deep_auto_regressive.core.rnn_model.ARModel2] from explicit hyperparameters.

    Args:
        n_locations: Number of locations (passed to the preprocess stage).
        cell: Recurrent cell type, a key of
            [`CELL_TYPES`][chapkit_deep_auto_regressive.core.rnn_model.CELL_TYPES] (``"gru"`` or ``"simple"``).
        rnn_features: Hidden width of the encoder and decoder cells.
        preprocess_hidden: Hidden width of the per-location preprocess stack.
        preprocess_output: Output width of the preprocess stage.
        embedding_dim: Size of the per-location embedding.
        head_features: Hidden width of the output head.
        rnn_layers: Number of stacked encoder/decoder RNN layers.
        recursive_decode: Whether to decode the horizon one step at a time, feeding
            each step's predicted mean back as the next step's auto-regressive input.
        dropout_rate: Dropout probability in the preprocess stage.
        input_dropout_rate: Feature-dropout probability on the raw covariates,
            zeroing whole feature channels (shared across location and time).

    Returns:
        The configured network.
    """
    cell_cls = CELL_TYPES[cell]
    return ARModel2(
        Preprocess(
            n_locations=n_locations,
            n_hidden=preprocess_hidden,
            embedding_dim=embedding_dim,
            output_dim=preprocess_output,
            dropout_rate=dropout_rate,
            input_dropout_rate=input_dropout_rate,
        ),
        cell_cls(features=rnn_features),
        cell_cls(features=rnn_features),
        head_features=head_features,
        n_layers=rnn_layers,
        recursive_decode=recursive_decode,
    )


def _multi_value_network(n_locations: int) -> ARModel2:
    """A small network with the cross-location auto-regressive join."""
    return ARModel2(
        Preprocess(n_locations=n_locations, output_dim=2, dropout_rate=0.2),
        nn.SimpleCell(features=4),
        nn.SimpleCell(features=4),
        ar_adder=MultiValueARAdder(),
    )


#: Named network factories. ``"base"`` builds the default network (GRU-based);
#: ``"multi_value"`` swaps in the cross-location auto-regressive join.
model_makers: dict[str, Callable[[int], ARModel2]] = {
    "base": build_network,
    "multi_value": _multi_value_network,
}
