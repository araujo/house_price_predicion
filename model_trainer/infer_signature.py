"""MLflow model signatures from sample predictions (serving contract)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.models.signature import ModelSignature, infer_signature


def infer_signature_from_predictions(
    X: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
) -> ModelSignature:
    """
    Build an MLflow signature from feature rows and model outputs.

    ``predictions`` must be on the same scale as logged metrics (e.g. price in dollars).
    """
    y = np.asarray(predictions).ravel()
    return infer_signature(X, y)
