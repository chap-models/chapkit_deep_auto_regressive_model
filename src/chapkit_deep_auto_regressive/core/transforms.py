"""Feature extraction and scaling.

The model consumes a single tidy :class:`pandas.DataFrame` with one row per
location and time period and the columns ``location``, ``time_period``,
``rainfall``, ``mean_temperature``, ``population`` and (for training) the target
``disease_cases``.

- [`get_series`][chapkit_deep_auto_regressive.core.transforms.get_series] turns that frame into the dense
  ``(features, target)`` arrays the network consumes.
- [`ZScaler`][chapkit_deep_auto_regressive.core.transforms.ZScaler] standardizes those features so that no
  single covariate dominates training.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data_loader import Array, FeaturePair, Features


class FeatureSource(Protocol):
    """Anything that exposes its feature array as ``predictors(0)``, such as a DataSet."""

    def predictors(self, i: int) -> Features:
        """Return array ``i``; ``0`` is the features."""
        ...


#: The covariates every configuration must provide. They are always the leading
#: features, in this order; additional covariates are appended after them.
REQUIRED_COVARIATES: tuple[str, ...] = ("rainfall", "mean_temperature", "population")


@dataclass
class ZScaler:
    """Standardizes features to zero mean and unit variance.

    The scaler stores the per-feature mean and standard deviation estimated from
    the training data and applies ``(x - mu) / std`` when called. Standardizing
    keeps features such as rainfall (hundreds of millimetres) and the day-of-year
    position (between 0 and 1) on a comparable scale, which helps the optimizer.

    Attributes:
        mu: Per-feature means, broadcast over locations and time.
        std: Per-feature standard deviations, broadcast over locations and time.
    """

    mu: Array
    std: Array

    def __call__(self, x: FeaturePair) -> FeaturePair:
        """Standardize the feature array in a ``(features, ar_target)`` pair.

        Args:
            x: The features to scale and the auto-regressive target, which is
                passed through unchanged.

        Returns:
            The same pair with its features standardized.
        """
        features, ar_target = x
        return (features - self.mu) / self.std, ar_target

    @classmethod
    def from_data(cls, data_set: FeatureSource) -> "ZScaler":
        """Fit a scaler from a dataset's feature statistics.

        Args:
            data_set: A [`DataSet`][chapkit_deep_auto_regressive.core.data_loader.DataSet] whose
                ``predictors(0)`` array provides the features.

        Returns:
            A ``ZScaler`` holding the mean and standard deviation of those
            features over the location and time axes.
        """
        features = data_set.predictors(0)
        mu = np.mean(features, axis=(0, 1))
        std = np.std(features, axis=(0, 1))
        # A zero-variance feature (a single location, or a static covariate such
        # as a constant population) would otherwise divide by zero and produce
        # inf/NaN inputs. Treat its scale as 1 so it standardizes to 0.
        std = np.where(std == 0.0, 1.0, std)
        return ZScaler(mu, std)


def location_groups(data: pd.DataFrame) -> Iterator[tuple[Any, pd.DataFrame]]:
    """Yield each location's rows, sorted by time period.

    Locations are yielded in sorted (canonical) label order so that a location
    maps to the same embedding index regardless of the input row order — this
    keeps ``train`` and ``predict`` aligned. Within each location the rows are
    sorted by the start date of their ``time_period``. Text order would do for
    ``YYYY-MM`` and ``start/end``, but not for ISO weeks such as ``2020W2`` /
    ``2020W10`` or mixed ``2020W01`` / ``2020-W02``.

    Args:
        data: The input frame with a ``location`` and ``time_period`` column.

    Yields:
        ``(location, sub_frame)`` pairs.
    """
    for location, sub in data.groupby("location", sort=True):
        starts = [period_start_date(period) for period in sub["time_period"]]
        order = sorted(range(len(starts)), key=starts.__getitem__)  # stable, so ties keep their order
        yield location, sub.iloc[order]


def _check_consecutive(location: Any, periods: Sequence[Any]) -> None:
    """Raise unless ``periods`` (sorted) are consecutive months or weeks, with no duplicates or gaps."""
    starts = [period_start_date(period) for period in periods]
    for i in range(1, len(starts)):
        before, after = starts[i - 1], starts[i]
        if re.fullmatch(r"\d{4}-\d{2}", str(periods[i - 1])):
            consecutive = (after.year * 12 + after.month) - (before.year * 12 + before.month) == 1
        else:
            consecutive = (after - before).days == 7
        if not consecutive:
            raise ValueError(
                f"location {location!r}: time periods must be consecutive, with no duplicates or gaps, "
                f"but {periods[i - 1]!r} is followed by {periods[i]!r}"
            )


def get_series(data: pd.DataFrame, covariates: Sequence[str] = REQUIRED_COVARIATES) -> tuple[Array, Array]:
    """Extract dense feature and target arrays from the input frame.

    For every location the function stacks one feature per ``covariates`` column
    (rainfall, mean temperature and population by default), followed by the
    day-of-year position, into a ``(periods, features)`` matrix, and collects the
    observed ``disease_cases`` as the target when the column is present. Passing
    extra names in ``covariates`` simply appends them as additional features; the
    network infers its input width from the data, so no other change is needed.

    Args:
        data: A tidy frame with one row per location and time period.
        covariates: The covariate columns to use as features, in order. Defaults
            to the three required covariates; additional names are appended after
            them (the ``year_position`` feature always comes last).

    Returns:
        A ``(x, y)`` tuple where ``x`` has shape
        ``(locations, periods, len(covariates) + 1)`` and ``y`` has shape
        ``(locations, periods)``. ``y`` is empty when the frame carries no
        ``disease_cases`` column (i.e. future data).

    Raises:
        ValueError: If a requested covariate column is missing, if a location's
            periods are not consecutive (duplicates or gaps), if the locations
            do not all share the same number of periods (the dense array is
            rectangular by construction), if any feature value is NaN or
            infinite, or if ``disease_cases`` is negative or infinite.
    """
    has_target = "disease_cases" in data.columns
    missing = [c for c in covariates if c not in data.columns]
    if missing:
        raise ValueError(f"missing covariate column(s) {missing}; available columns: {list(data.columns)}")
    xs: list[Array] = []
    ys: list[Array] = []
    counts: dict[Any, int] = {}
    for location, sub in location_groups(data):
        counts[location] = len(sub)
        _check_consecutive(location, sub["time_period"].tolist())
        yp = np.asarray([year_position_from_period(period) for period in sub["time_period"]])
        features: list[Array] = [sub[name].to_numpy() for name in covariates]
        # Seasonal calendar features: the year-position ramp plus the first two
        # Fourier harmonics, which give the network a smooth, periodic seasonal
        # signal (a single ramp cannot express the seasonal shape on its own).
        features.append(yp)
        features.append(np.sin(2 * np.pi * yp))
        features.append(np.cos(2 * np.pi * yp))
        features.append(np.sin(4 * np.pi * yp))
        features.append(np.cos(4 * np.pi * yp))
        xs.append(np.array(features).T)
        if has_target:
            cases = sub["disease_cases"].to_numpy()
            numeric = cases.astype(float)
            if np.any(numeric < 0) or np.any(np.isinf(numeric)):
                raise ValueError(f"location {location!r}: disease_cases must be finite and not negative")
            ys.append(cases)
    if len(set(counts.values())) > 1:
        raise ValueError(f"every location must have the same number of periods, but the period counts differ: {counts}")
    x = np.array(xs)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"input features contain NaN or infinite values (one of {tuple(covariates)})")
    return x, np.array(ys)


def period_start_date(period: str) -> datetime:
    """Return the start date of a CHAP time-period string.

    Args:
        period: A monthly period (``"YYYY-MM"``), a weekly range
            (``"YYYY-MM-DD/YYYY-MM-DD"``) or an ISO week (``"YYYYWww"`` or
            ``"YYYY-Www"``, starting on its Monday).

    Returns:
        The first day of the period as a ``datetime``.
    """
    text = str(period)
    if "/" in text:
        start_text, end_text = text.split("/", 1)
        start, end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
        if (end - start).days != 6:
            raise ValueError(f"time_period {text!r} is not a week: a start/end range must span 7 days")
        return start
    # ISO weeks, the week format chapkit sends.
    iso_week = re.fullmatch(r"(\d{4})-?W(\d{1,2})", text)
    if iso_week:
        return datetime.fromisocalendar(int(iso_week[1]), int(iso_week[2]), 1)
    return datetime.strptime(text, "%Y-%m")


def year_position_from_period(period: str) -> float:
    """Return the within-year position of a period start as a fraction in ``[0, 1]``.

    This gives the network a simple, continuous seasonal signal: January is near 0
    and December is near 1.

    Args:
        period: A CHAP time-period string.

    Returns:
        The day of the year of the period's start divided by 365.
    """
    return period_start_date(period).timetuple().tm_yday / 365
