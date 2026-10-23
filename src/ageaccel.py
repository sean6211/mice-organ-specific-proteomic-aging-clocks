"""Horvath AgeAccel.l residuals and within-mouse organ asynchrony."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ORGAN_ORDER = ["kidney", "spleen", "liver", "lung", "heart", "brain", "testis", "muscle"]
N_PERM = 10_000
RNG = np.random.default_rng(0)


def ageaccel_l(predicted: np.ndarray, chronological: np.ndarray) -> np.ndarray:
    """Horvath 2013 AgeAccelerationResidual: residual of predicted ~ chronological age."""
    predicted = np.asarray(predicted, dtype=float)
    chronological = np.asarray(chronological, dtype=float)
    x = np.column_stack([np.ones(len(chronological)), chronological])
    beta, *_ = np.linalg.lstsq(x, predicted, rcond=None)
    return predicted - x @ beta


def add_ageaccel(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["mouse_id"] = out["mouse_id"].astype(int)
    out["age_months"] = out["age_months"].astype(float)
    out["ageaccel_l"] = np.nan
    for organ, sub in out.groupby("organ"):
        idx = sub.index
        out.loc[idx, "ageaccel_l"] = ageaccel_l(
            sub["pred_loo"].to_numpy(), sub["age_months"].to_numpy()
        )
    return out


def complete_case_mice(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.dropna(axis=0, how="any")


def organ_correlation(ageaccel: pd.DataFrame) -> pd.DataFrame:
    organs = [organ for organ in ORGAN_ORDER if organ in ageaccel["organ"].unique()]
    wide = (
        ageaccel.pivot(index="mouse_id", columns="organ", values="ageaccel_l")
        .reindex(columns=organs)
    )
    wide = complete_case_mice(wide)
    return wide.corr(method="spearman")


def asynchrony_scores(ageaccel: pd.DataFrame) -> pd.DataFrame:
    organs = [organ for organ in ORGAN_ORDER if organ in ageaccel["organ"].unique()]
    wide = (
        ageaccel.pivot(index="mouse_id", columns="organ", values="ageaccel_l")
        .reindex(columns=organs)
    )
    ages = (
        ageaccel.drop_duplicates("mouse_id")
        .set_index("mouse_id")["age_months"]
        .reindex(wide.index)
    )
    complete = wide.notna().all(axis=1)
    mad = wide.apply(lambda row: np.nanmedian(np.abs(row - np.nanmedian(row))), axis=1)
    sd = wide.std(axis=1, skipna=True)
    scores = pd.DataFrame(
        {
            "mouse_id": wide.index.astype(int),
            "age_months": ages.to_numpy(),
            "asynchrony_mad": mad.to_numpy(),
            "asynchrony_sd": sd.to_numpy(),
            "n_organs": wide.notna().sum(axis=1).to_numpy(),
            "complete_case": complete.to_numpy(),
        }
    )
    return scores.sort_values(["age_months", "mouse_id"]).reset_index(drop=True)


def _spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM) -> tuple[float, float, float]:
    r_obs, p_param = _spearman_p(x, y)
    if not np.isfinite(r_obs):
        return r_obs, p_param, float("nan")
    count = 0
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    for _ in range(n_perm):
        r_perm, _ = _spearman_p(x, RNG.permutation(y))
        if np.isfinite(r_perm) and abs(r_perm) >= abs(r_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return r_obs, p_param, float(p_perm)


def permutation_mannwhitney(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan"), float("nan")
    stat = mannwhitneyu(a, b, alternative="two-sided")
    combined = np.concatenate([a, b])
    n_a = len(a)
    n_b = len(b)
    obs = stat.statistic
    null_mean = n_a * n_b / 2.0
    obs_dev = abs(obs - null_mean)
    count = 0
    work = combined.copy()
    for _ in range(n_perm):
        RNG.shuffle(work)
        stat_p = mannwhitneyu(work[:n_a], work[n_a:], alternative="two-sided").statistic
        if abs(stat_p - null_mean) >= obs_dev:
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return float(stat.statistic), float(stat.pvalue), float(p_perm)


def asynchrony_tests(scores: pd.DataFrame) -> pd.DataFrame:
    complete = scores.loc[scores["complete_case"]].copy()
    r, p_param, p_perm = permutation_spearman(
        complete["age_months"].to_numpy(), complete["asynchrony_mad"].to_numpy()
    )
    young = complete.loc[complete["age_months"] <= 14, "asynchrony_mad"].to_numpy()
    old = complete.loc[complete["age_months"] >= 20, "asynchrony_mad"].to_numpy()
    mw_stat, mw_p, mw_p_perm = permutation_mannwhitney(old, young)
    return pd.DataFrame(
        [
            {
                "test": "spearman_asynchrony_vs_age",
                "subset": "complete_case",
                "n": int(len(complete)),
                "estimate": r,
                "p_parametric": p_param,
                "p_permutation": p_perm,
            },
            {
                "test": "mannwhitney_20_26_vs_3_14",
                "subset": "complete_case",
                "n": int(len(young) + len(old)),
                "estimate": float(np.nanmedian(old) - np.nanmedian(young)),
                "p_parametric": mw_p,
                "p_permutation": mw_p_perm,
            },
        ]
    )


def ageaccel_wide(ageaccel: pd.DataFrame) -> pd.DataFrame:
    organs = [organ for organ in ORGAN_ORDER if organ in ageaccel["organ"].unique()]
    ages = ageaccel.drop_duplicates("mouse_id").set_index("mouse_id")["age_months"]
    wide = (
        ageaccel.pivot(index="mouse_id", columns="organ", values="ageaccel_l")
        .reindex(columns=organs)
    )
    wide.insert(0, "age_months", ages.reindex(wide.index).to_numpy())
    wide.index.name = "mouse_id"
    return wide.sort_values(["age_months", "mouse_id"])


def ageaccel_pca(ageaccel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    organs = [organ for organ in ORGAN_ORDER if organ in ageaccel["organ"].unique()]
    wide = (
        ageaccel.pivot(index="mouse_id", columns="organ", values="ageaccel_l")
        .reindex(columns=organs)
    )
    wide = complete_case_mice(wide)
    ages = (
        ageaccel.drop_duplicates("mouse_id")
        .set_index("mouse_id")["age_months"]
        .reindex(wide.index)
    )
    x = wide.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = pd.DataFrame(
        {
            "mouse_id": wide.index.astype(int),
            "age_months": ages.to_numpy(),
            "pc1": u[:, 0] * s[0],
            "pc2": u[:, 1] * s[1],
        }
    )
    mean_accel = wide.mean(axis=1).to_numpy()
    if np.corrcoef(scores["pc1"], mean_accel)[0, 1] < 0:
        scores["pc1"] *= -1
        vt[0] *= -1
    var_frac = (s**2) / np.sum(s**2)
    scores["pc1_var_frac"] = var_frac[0]
    scores["pc2_var_frac"] = var_frac[1]
    loadings = pd.DataFrame({"organ": organs, "pc1": vt[0], "pc2": vt[1]})
    variance = pd.DataFrame(
        {"pc": [f"pc{i+1}" for i in range(len(s))], "variance_frac": var_frac}
    )
    return scores, loadings, variance


def write_asynchrony(predictions: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    async_dir = out_dir / "asynchrony"
    async_dir.mkdir(parents=True, exist_ok=True)
    ageaccel = add_ageaccel(predictions)
    corr = organ_correlation(ageaccel)
    scores = asynchrony_scores(ageaccel)
    tests = asynchrony_tests(scores)
    wide = ageaccel_wide(ageaccel)
    pca_scores, pca_loadings, pca_var = ageaccel_pca(ageaccel)
    ageaccel.to_csv(async_dir / "ageaccel_l.csv", index=False)
    corr.to_csv(async_dir / "organ_gap_correlation.csv")
    scores.to_csv(async_dir / "asynchrony_scores.csv", index=False)
    tests.to_csv(async_dir / "asynchrony_tests.csv", index=False)
    wide.to_csv(async_dir / "ageaccel_wide.csv")
    pca_scores.to_csv(async_dir / "pca_scores.csv", index=False)
    pca_loadings.to_csv(async_dir / "pca_loadings.csv", index=False)
    pca_var.to_csv(async_dir / "pca_variance.csv", index=False)
    return ageaccel, corr, scores, tests
