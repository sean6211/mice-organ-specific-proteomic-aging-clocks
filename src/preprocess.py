"""Filter, log2-transform, and median-normalize organ LFQ matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OBSERVED_FRAC = 0.80


def abundance_block(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.drop(columns=["description"], errors="ignore")
    values.columns = values.columns.map(str)
    return values.apply(pd.to_numeric, errors="coerce")


def filter_complete(matrix: pd.DataFrame, min_frac: float = MIN_OBSERVED_FRAC) -> pd.DataFrame:
    values = abundance_block(matrix)
    keep = values.notna().mean(axis=1) >= min_frac
    return matrix.loc[keep].copy()


def log2_median_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    """log2 LFQ then subtract per-sample nanmedian. Missing values stay NaN."""
    values = abundance_block(matrix)
    logged = np.log2(values)
    normalized = logged.sub(logged.median(axis=0, skipna=True), axis=1)
    out = matrix.copy()
    out.loc[:, normalized.columns] = normalized
    return out


def preprocess_organ(matrix: pd.DataFrame, min_frac: float = MIN_OBSERVED_FRAC) -> pd.DataFrame:
    filtered = filter_complete(matrix, min_frac=min_frac)
    return log2_median_normalize(filtered)
