"""The deep auto-regressive (RNN) flax model for Chap.

Vendored from https://github.com/chap-models/chap_auto_regressive at commit
c4d1d5c (main, 2026-11-10). Local changes: the package name in imports and
docstring cross-references, and ISO-week periods ("2020W05", "2020-W05") in
transforms.period_start_date. The library tests are in tests/library/, unchanged
but for the package name.
"""

from .model import AutoRegressiveModel

__all__ = ["AutoRegressiveModel"]
