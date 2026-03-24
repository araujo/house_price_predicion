"""Train / validation splits for regression (deterministic)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Random train/validation split with fixed seed for reproducibility.

    Parameters
    ----------
    X, y:
        Aligned features and target (``price``).
    test_size:
        Fraction held out for validation.
    random_state:
        Seed passed to sklearn (training/serving experiments stay comparable).
    shuffle:
        Whether to shuffle before splitting (default True).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
