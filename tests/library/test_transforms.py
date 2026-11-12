import numpy as np
import pandas as pd
import pytest

from chapkit_auto_regressive.core.transforms import ZScaler, get_series, period_start_date, year_position_from_period


def _rows(location, periods):
    return [
        {
            "location": location,
            "time_period": p,
            "rainfall": 1.0,
            "mean_temperature": 20.0,
            "population": 1000.0,
            "disease_cases": 1.0,
        }
        for p in periods
    ]


def test_get_series_rejects_ragged_period_counts():
    # Public dataframe input with differing period counts per location must raise
    # a targeted error naming the counts, not a low-level NumPy array error.
    ragged = pd.DataFrame(_rows("A", ["2020-01", "2020-02", "2020-03"]) + _rows("B", ["2020-01", "2020-02"]))
    with pytest.raises(ValueError, match="same number of periods"):
        get_series(ragged)


def test_zscaler_standardizes_first_element():
    mu = np.array([1.0, 2.0])
    std = np.array([2.0, 4.0])
    feats = np.array([[3.0, 6.0]])
    out = ZScaler(mu, std)((feats,))
    assert np.allclose(out[0], (feats - mu) / std)


def test_year_position_bounds():
    assert year_position_from_period("2024-01") < 0.02
    assert year_position_from_period("2024-12") > 0.9


def test_period_start_date_monthly_and_weekly():
    assert period_start_date("2010-03").month == 3
    weekly = period_start_date("2003-12-29/2004-01-04")
    assert (weekly.year, weekly.month, weekly.day) == (2003, 12, 29)


class _Predictors:
    """Minimal stand-in exposing ``predictors(0)`` for ``ZScaler.from_data``."""

    def __init__(self, features):
        self._features = features

    def predictors(self, i):
        return self._features


def test_zscaler_handles_zero_variance_features():
    # A constant feature (single location or static covariate) has std 0; the
    # scaler must not divide by zero and produce inf/NaN inputs.
    features = np.array([[[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]])  # second feature constant
    scaler = ZScaler.from_data(_Predictors(features))
    assert scaler.std[1] == 1.0  # zero variance floored to 1
    scaled = scaler((features,))[0]
    assert np.isfinite(scaled).all()
    assert np.allclose(scaled[..., 1], 0.0)  # constant feature standardizes to 0
