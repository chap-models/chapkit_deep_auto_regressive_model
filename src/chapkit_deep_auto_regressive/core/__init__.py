"""The network, its training loop and the data handling behind the model.

- ``rnn_model``: the encoder-decoder recurrent network.
- ``trainer``: the Adam training loop, with optional early stopping.
- ``data_loader`` and ``transforms``: windowing, feature extraction and scaling.
- ``distributions``: the negative binomial output head.
- ``model``: ``DeepAutoRegressiveModel``, which trains a ``FlaxPredictor``.
- ``options``: ``ModelOptions``, every knob with its default, as a pydantic model.
"""

from .model import DeepAutoRegressiveModel, FlaxPredictor
from .options import ModelOptions

__all__ = ["DeepAutoRegressiveModel", "FlaxPredictor", "ModelOptions"]
