import numpy as np

from chapkit_deep_auto_regressive.core.data_loader import DataSet, SimpleDataLoader, interpolate_nans


def test_interpolate_nans_fills_linearly():
    out = interpolate_nans(np.array([[1.0, np.nan, 3.0]]))
    assert not np.isnan(out).any()
    assert out[0, 1] == 2.0


def test_interpolate_nans_all_nan_row_does_not_crash():
    # A location with no observed cases at all used to crash np.interp with
    # "array of sample points is empty"; it must now fill with finite zeros.
    out = interpolate_nans(np.array([[np.nan, np.nan, np.nan], [1.0, np.nan, 3.0]]))
    assert np.isfinite(out).all()
    assert np.array_equal(out[0], [0.0, 0.0, 0.0])
    assert out[1, 1] == 2.0  # the other row still interpolates normally


def test_dataset_window_shapes():
    X = np.zeros((2, 10, 4), dtype="float32")
    y = np.ones((2, 10), dtype="float32")
    ds = DataSet(X, y, forecast_length=2, context_length=4)

    assert len(ds) == 10 - 6 + 1  # windows of length context + forecast = 6
    x, ar_y, full_y = ds[0]
    assert x.shape == (2, 6, 4)
    assert ar_y.shape == (2, 4)
    assert full_y.shape == (2, 6)


def test_simple_data_loader_walks_all_windows():
    X = np.zeros((1, 8, 3), dtype="float32")
    y = np.ones((1, 8), dtype="float32")
    ds = DataSet(X, y, forecast_length=2, context_length=3)
    assert len(list(SimpleDataLoader(ds))) == len(ds)
