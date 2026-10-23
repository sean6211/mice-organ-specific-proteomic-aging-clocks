"""Steiger test for two correlations that share one variable."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def steiger_z(r_xy: float, r_xz: float, r_yz: float, n: int) -> tuple[float, float]:
    """Two-tailed Steiger test of r(x,y) vs r(x,z) with overlapping x."""
    if min(n, 4) < 4 or not np.isfinite(r_xy) or not np.isfinite(r_xz) or not np.isfinite(r_yz):
        return np.nan, np.nan
    rm2 = (r_xy**2 + r_xz**2) / 2.0
    f = (1.0 - r_yz) / (2.0 * (1.0 - rm2)) if abs(1.0 - rm2) > 1e-12 else np.nan
    h = (1.0 - f * rm2) / (1.0 - rm2) if np.isfinite(f) and abs(1.0 - rm2) > 1e-12 else np.nan
    if not np.isfinite(h) or h <= 0:
        return np.nan, np.nan
    zxy = np.arctanh(np.clip(r_xy, -0.999999, 0.999999))
    zxz = np.arctanh(np.clip(r_xz, -0.999999, 0.999999))
    denom = np.sqrt(2.0 * h * (1.0 - r_yz))
    if denom < 1e-12:
        return np.nan, np.nan
    z = (zxy - zxz) * np.sqrt(n - 3) / denom
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(z), float(p)
