"""Moderated F-tests for age-associated proteins on clock LFQ matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, false_discovery_control

from .preprocess import abundance_block, preprocess_organ


def _design_matrix(ages: np.ndarray) -> tuple[np.ndarray, int]:
    levels = np.sort(np.unique(ages))
    dummy = (ages[:, None] == levels[None, :]).astype(float)
    return dummy, dummy.shape[1]


def _moderated_f(values: pd.DataFrame, ages: np.ndarray) -> pd.DataFrame:
    x, p = _design_matrix(ages)
    n = x.shape[0]
    df_res = n - p
    if df_res < 1:
        raise ValueError("Not enough samples for an age-factor F-test")
    xtx_inv = np.linalg.pinv(x.T @ x)
    hat = xtx_inv @ x.T
    y = values.to_numpy(dtype=float)
    missing = ~np.isfinite(y)
    y_filled = np.where(missing, np.nanmean(y, axis=1, keepdims=True), y)
    beta = hat @ y_filled.T
    fitted = x @ beta
    resid = y_filled.T - fitted
    rss = np.sum(resid**2, axis=0)
    s2 = rss / df_res
    finite = np.isfinite(s2) & (s2 > 0)
    s2_prior = float(np.median(s2[finite])) if finite.any() else 1.0
    df_prior = 4.0
    s2_mod = (df_res * np.where(finite, s2, s2_prior) + df_prior * s2_prior) / (df_res + df_prior)
    ybar = np.mean(y_filled, axis=1, keepdims=True)
    tss = np.sum((y_filled - ybar) ** 2, axis=1)
    msb = np.maximum(tss - rss, 0.0) / (p - 1)
    f_stat = msb / s2_mod
    pval = f_dist.sf(f_stat, p - 1, df_res + df_prior)
    pval = np.where(np.isfinite(pval), pval, 1.0)
    qval = false_discovery_control(pval, method="bh")
    return pd.DataFrame(
        {
            "accession": values.index,
            "f_stat": f_stat,
            "p_value": pval,
            "q_value": qval,
            "s2": s2,
            "s2_moderated": s2_mod,
        }
    )


def age_de_organ(organ: str, matrix: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    processed = preprocess_organ(matrix)
    organ_samples = samples.loc[samples["organ"] == organ].copy()
    organ_samples["mouse_id"] = organ_samples["mouse_id"].astype(str)
    values = abundance_block(processed)
    cols = [mid for mid in organ_samples["mouse_id"].tolist() if mid in values.columns]
    values = values.reindex(columns=cols)
    ages = (
        organ_samples.set_index("mouse_id")
        .loc[cols, "age_months"]
        .to_numpy(dtype=float)
    )
    result = _moderated_f(values, ages)
    result.insert(0, "organ", organ)
    if "description" in processed.columns:
        result["description"] = processed["description"].reindex(result["accession"]).to_numpy()
    result["age_de"] = result["q_value"] < 0.05
    return result.sort_values(["q_value", "p_value"])


def overlap_sets(
    de: pd.DataFrame,
    coefficients: pd.DataFrame,
    stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    de_set = set(de.loc[de["age_de"], "accession"])
    clock_set = set(coefficients["accession"])
    stable_set = set(stability.loc[stability["selection_frequency"] > 0.5, "accession"])
    organ = str(de["organ"].iloc[0])
    summary = pd.DataFrame(
        [
            {
                "organ": organ,
                "n_tested": int(len(de)),
                "n_age_de": int(len(de_set)),
                "n_full_clock": int(len(clock_set)),
                "n_stable_clock": int(len(stable_set)),
                "jaccard_de_clock": _jaccard(de_set, clock_set),
                "jaccard_de_stable": _jaccard(de_set, stable_set),
                "jaccard_clock_stable": _jaccard(clock_set, stable_set),
                "n_de_and_clock": int(len(de_set & clock_set)),
                "n_de_and_stable": int(len(de_set & stable_set)),
                "n_clock_and_stable": int(len(clock_set & stable_set)),
            }
        ]
    )
    rows = []
    universe = de_set | clock_set | stable_set
    for acc in sorted(universe):
        rows.append(
            {
                "organ": organ,
                "accession": acc,
                "age_de": acc in de_set,
                "full_clock": acc in clock_set,
                "stable_clock": acc in stable_set,
            }
        )
    return summary, pd.DataFrame(rows)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def write_de_overlap(
    matrices: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    coef_dir: Path,
    stab_dir: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    de_dir = out_dir / "de_overlap"
    de_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    memberships = []
    for organ, matrix in matrices.items():
        print(f"Age DE F-test {organ} ...", flush=True)
        de = age_de_organ(organ, matrix, samples)
        de.to_csv(de_dir / f"{organ}_de.csv", index=False)
        coefs = pd.read_csv(coef_dir / f"{organ}.csv")
        stability = pd.read_csv(stab_dir / f"{organ}.csv")
        summary, members = overlap_sets(de, coefs, stability)
        summaries.append(summary)
        memberships.append(members)
        print(
            f"  {organ}: {int(summary.iloc[0]['n_age_de'])} DE, "
            f"{int(summary.iloc[0]['n_full_clock'])} full-fit clock, "
            f"{int(summary.iloc[0]['n_stable_clock'])} stable clock",
            flush=True,
        )
    summary = pd.concat(summaries, ignore_index=True)
    membership = pd.concat(memberships, ignore_index=True)
    summary.to_csv(de_dir / "set_overlap_summary.csv", index=False)
    membership.to_csv(de_dir / "set_membership.csv", index=False)
    return summary, membership
