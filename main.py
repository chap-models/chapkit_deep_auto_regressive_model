"""Service entry point: ``uvicorn main:app``.

The service is defined in ``chapkit_deep_auto_regressive.service``. This module
re-exports it at the project root, next to the MLproject, so that
``chapkit mlproject export`` (``make mlproject``) reads it from here and writes
a portable "Generated from main.py" header.
"""

from chapkit_deep_auto_regressive.service import HIERARCHY, AutoRegressiveConfig, app, info, runner

__all__ = ["HIERARCHY", "AutoRegressiveConfig", "app", "info", "runner"]

if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app", reload=False)
