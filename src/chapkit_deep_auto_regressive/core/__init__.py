"""The deep auto-regressive (RNN) flax model for Chap.

Vendored from https://github.com/chap-models/chap_auto_regressive at commit
c4d1d5c (main, 2026-11-10). Local changes: the package name in imports and
docstring cross-references, and three changes that keep the forecasts of the
weekly and monthly v2 models and the weeks chapkit sends:

- ISO-week periods ("2020W05", "2020-W05") in transforms.period_start_date.
- transforms.location_groups orders each location's rows by period start date,
  not by text (the same order for "YYYY-MM" and "start/end", the only formats
  the library read before; the right order for ISO weeks).
- With a learning-rate schedule and early stopping, AutoRegressiveModel.train
  sizes the probe's schedule by the full series, as the weekly v2 model did, not
  by the probe's own windows.

The library tests are in tests/library/, unchanged but for the package name.
"""

from .model import AutoRegressiveModel

__all__ = ["AutoRegressiveModel"]
