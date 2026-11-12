"""Predict entry point.

Usage: ``python -m chapkit_deep_auto_regressive.predict <model> <historic.csv> <future.csv> <out.csv>``.

The trained model file carries the architecture, window lengths and covariates,
so predict needs no configuration and works the same for weekly and monthly data.
"""

import argparse
import logging

import pandas as pd

from chapkit_deep_auto_regressive.core.model import FlaxPredictor

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Load a trained model and write forecast samples to a CSV."""
    parser = argparse.ArgumentParser(description="Forecast with the deep auto-regressive model")
    parser.add_argument("model", help="path to the trained model")
    parser.add_argument("historic_data", help="path to the historic data CSV (with disease_cases)")
    parser.add_argument("future_data", help="path to the future covariates CSV (no disease_cases)")
    parser.add_argument("out_file", help="path to write the predictions CSV")
    args = parser.parse_args()

    predictor = FlaxPredictor.load(args.model)
    forecasts = predictor.predict(pd.read_csv(args.historic_data), pd.read_csv(args.future_data))
    forecasts.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()
