#!/usr/bin/env python3
"""
Legacy baseline parity script (typical ``create_model.py`` name in KC house tutorials).

Trains only the **baseline_knn** benchmark (``KNeighborsRegressor`` + shared Phase 3
preprocessing). For the full comparison grid, use::

    python -m model_trainer.train

Or::

    python -m model_trainer.train --models baseline_knn --no-mlflow
"""

from __future__ import annotations

from model_trainer.train import main as train_main


def main_baseline() -> int:
    """Run training restricted to ``baseline_knn`` for apples-to-apples baseline runs."""
    return train_main(["--models", "baseline_knn"])


if __name__ == "__main__":
    raise SystemExit(main_baseline())
