"""Windowing of time series into training instances.

The model is trained on fixed-length windows of ``context_length + forecast_length``
periods sliced out of each location's series. The classes here turn the dense
``(features, target)`` arrays from [`transforms`][chapkit_auto_regressive.core.transforms] into those
windows and iterate over them.

Each window yields three arrays:

- ``x`` — features over the whole window (context + forecast),
- ``ar_y`` — the (NaN-interpolated) observed target over the **context** only,
  which is fed back in as the auto-regressive input,
- ``y`` — the raw target over the whole window, used as the training label.
"""

from typing import Callable, Iterable, Tuple

import numpy as np


def nan_helper(y: np.ndarray) -> tuple:
    """Return helpers for locating NaNs in a 1-D array.

    Args:
        y: A 1-D array.

    Returns:
        A tuple ``(mask, index_fn)`` where ``mask`` is a boolean array marking the
        NaNs and ``index_fn`` maps such a mask to the integer positions it
        selects — the form expected by ``numpy.interp``.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaNs in each row of a 2-D array.

    Gaps in surveillance leave NaNs in the case series. Because the past target is
    fed back into the network as an input, those gaps are filled by linear
    interpolation here (the likelihood still skips the originally-missing periods,
    so the gaps do not contribute a spurious training signal).

    Args:
        y: A ``(locations, periods)`` array, possibly containing NaNs.

    Returns:
        A copy of ``y`` with NaNs replaced by linear interpolation within each row.
    """
    y = y.copy()
    for row in y:
        nans, x = nan_helper(row)
        if nans.all():
            # No observed cases at all for this location/window: there is nothing
            # to interpolate from. Fill the auto-regressive input with zeros so it
            # stays finite; the raw target keeps its NaNs, so the likelihood still
            # skips these periods and they contribute no training signal.
            row[nans] = 0.0
            continue
        row[nans] = np.interp(x(nans), x(~nans), row[~nans])
    return y


class DataSet:
    """A windowed view over one set of location series.

    Indexing returns a single training window; iterating a
    [`SimpleDataLoader`][chapkit_auto_regressive.core.data_loader.SimpleDataLoader] over the dataset
    walks every window. A feature transform (typically a
    [`ZScaler`][chapkit_auto_regressive.core.transforms.ZScaler]) can be attached and is applied lazily
    as windows are produced.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, forecast_length: int, context_length: int | None = None):
        """Store the series and pre-compute the NaN-interpolated target.

        Args:
            X: Features, shape ``(locations, periods, features)``.
            y: Target, shape ``(locations, periods)``.
            forecast_length: Number of periods to forecast per window.
            context_length: Number of history periods per window; defaults to all
                periods before the forecast region.
        """
        self._X = X
        self._y = y
        self._context_length = context_length or X.shape[1] - forecast_length
        self._forecast_length = forecast_length
        self._total_length = self._context_length + forecast_length
        self._interpolated_y = interpolate_nans(y)
        self._transform = lambda x: x

    def set_transform(self, transform: Callable) -> None:
        """Attach a feature transform applied to each produced window.

        Args:
            transform: A callable mapping a ``(features, ar_target)`` tuple to a
                transformed tuple, e.g. a [`ZScaler`][chapkit_auto_regressive.core.transforms.ZScaler].
        """
        self._transform = transform

    def predictors(self, i: int) -> np.ndarray:
        """Return one of the underlying arrays by index.

        Args:
            i: ``0`` for the features, ``1`` for the interpolated target, ``2``
                for the raw target.

        Returns:
            The selected array.
        """
        return (self._X, self._interpolated_y, self._y)[i]

    def __len__(self) -> int:
        """Return the number of windows that fit in the series."""
        return self._X.shape[1] - self._total_length + 1

    def __getitem__(self, item: int) -> tuple:
        """Return the training window starting at index ``item``.

        Args:
            item: The start offset of the window.

        Returns:
            A ``(x, ar_y, y)`` tuple: features over the full window, the
            interpolated target over the context, and the raw target over the
            full window — with the attached feature transform applied.
        """
        start = item
        return self._transform(
            (
                self._X[:, start : start + self._total_length],
                self._interpolated_y[:, start : start + self._context_length],
            )
        ) + (self._y[:, start : start + self._total_length],)

    def prediction_instance(self) -> tuple:
        """Return the most recent window, for forecasting.

        Returns:
            The ``(features, ar_target)`` pair for the last ``total_length``
            periods, with the feature transform applied — the input used to roll
            the model forward at prediction time.
        """
        return self._transform((self._X[:, -self._total_length :], self._interpolated_y[:, -self._context_length :]))


class SimpleDataLoader:
    """Iterates over every window of a [`DataSet`][chapkit_auto_regressive.core.data_loader.DataSet] in order."""

    def __init__(self, dataset: DataSet):
        """Wrap a dataset for iteration.

        Args:
            dataset: The dataset whose windows will be yielded.
        """
        self.dataset = dataset

    def __iter__(self):
        """Yield each window of the dataset in index order."""
        for i in range(len(self.dataset)):
            yield self.dataset[i]


class DataLoader:
    """Iterates over windows with an optional held-out validation block.

    Unlike [`SimpleDataLoader`][chapkit_auto_regressive.core.data_loader.SimpleDataLoader], this loader
    can carve a contiguous validation region out of the middle of the series and
    mask the overlapping windows out of training, so the same series provides both
    training and validation windows.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        forecast_length: int,
        context_length: int | None = None,
        do_validation: bool = False,
    ):
        """Build the windows and, optionally, the validation mask.

        Args:
            X: Features, shape ``(locations, periods, features)``.
            y: Target, shape ``(locations, periods)``.
            forecast_length: Number of periods to forecast per window.
            context_length: Number of history periods per window; defaults to all
                periods before the forecast region.
            do_validation: If true, reserve a window near the middle for
                validation and mask the windows that overlap it out of training.
        """
        self._X = X  # n_locations, n_periods, n_features
        self._y = y  # n_locations, n_periods
        self._interpolated_y = interpolate_nans(y)
        self._context_length = context_length or X.shape[1] - forecast_length
        self._forecast_length = forecast_length
        self._total_length = self._context_length + forecast_length
        self.validation_mask = np.ones(X.shape[1] - self._total_length + 1, dtype=bool)
        if do_validation:
            self._validation_index = (X.shape[1] - self._total_length) // 2
            self.validation_mask[
                self._validation_index - forecast_length : self._validation_index + forecast_length
            ] = False
        self.do_validation = do_validation

    def __len__(self) -> int:
        """Return the number of training windows (excluding any masked out)."""
        return np.sum(self.validation_mask)

    def __iter__(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield each non-masked training window as ``(x, ar_y, y)``."""
        starts = np.arange(self._X.shape[1] - self._total_length + 1)[self.validation_mask]
        permuted_starts = starts
        return (
            (
                self._X[:, start : start + self._total_length],
                self._interpolated_y[:, start : start + self._context_length],
                self._y[:, start : start + self._total_length],
            )
            for start in permuted_starts
        )

    def validation_set(self) -> tuple:
        """Return the held-out validation window as ``(x, ar_y, y)``.

        Returns:
            The single window reserved for validation. Only meaningful when the
            loader was created with ``do_validation=True``.
        """
        return (
            self._X[:, self._validation_index : self._validation_index + self._total_length],
            self._interpolated_y[:, self._validation_index : self._validation_index + self._context_length],
            self._y[:, self._validation_index : self._validation_index + self._total_length],
        )
