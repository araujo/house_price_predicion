"""Load raw CSV datasets for Seattle house price modeling."""

from pathlib import Path

import pandas as pd

from data_engineer.constants import (
    FUTURE_UNSEEN_FILENAME,
    KC_HOUSE_FILENAME,
    RAW_DATA_DIR,
    ZIPCODE_DEMOGRAPHICS_FILENAME,
)


def _resolve_raw_dir(raw_dir: Path | str | None) -> Path:
    if raw_dir is None:
        return RAW_DATA_DIR
    return Path(raw_dir)


def load_kc_house_dataframe(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Load King County house sales CSV."""
    path = _resolve_raw_dir(raw_dir) / KC_HOUSE_FILENAME
    return pd.read_csv(path)


def load_zipcode_demographics_dataframe(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Load zipcode-level demographics CSV."""
    path = _resolve_raw_dir(raw_dir) / ZIPCODE_DEMOGRAPHICS_FILENAME
    return pd.read_csv(path)


def load_future_unseen_examples_dataframe(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Load inference examples CSV (no id/date/price)."""
    path = _resolve_raw_dir(raw_dir) / FUTURE_UNSEEN_FILENAME
    return pd.read_csv(path)


def load_all_raw(
    raw_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, demographics, and inference CSVs."""
    base = _resolve_raw_dir(raw_dir)
    return (
        load_kc_house_dataframe(base),
        load_zipcode_demographics_dataframe(base),
        load_future_unseen_examples_dataframe(base),
    )
