"""Apply frozen Scifo liver clocks to Orwell / PXD035255 (no refit)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .burns_transfer import _accession, _group_contrast
from .takasugi_transfer import _frozen_weights, _zscore

ORGAN = "liver"
M001_MATRIX = "LC-M001-MouseLiverProtMatrix_raw_proteotypic.txt"
M001_META = "LC-M001.tsv"
M004_MATRIX = "LC-M004-Mouse_2019.12_Liver_protein_raw_proteotypic.txt"
M004_META = "LC-M004.tsv"
M001_COL_RE = re.compile(r"MouseLiver-(\d+)-GC(\d+)")
DIET_MAP = {
    "5LG6": "control",
    "Acarbose 1000 ppm": "acarbose",
    "Estradiol 14.4 ppm": "17aE2",
    "Rapamycin 14 ppm": "rapamycin",
}
M001_ARMS = ("control", "acarbose", "17aE2", "rapamycin")
M004_ARMS = ("control", "4EGI-1")
M004_TREATMENT_MAP = {
    "Control": "control",
    "4EGI-1": "4EGI-1",
}


def _drop_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in frame.columns:
        name = str(col).strip()
        if name == "" or name.lower().startswith("unnamed"):
            continue
        keep.append(col)
    return frame.loc[:, keep]


def _log2_protein_matrix(raw: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    raw = raw.copy()
    raw["accession"] = raw["ProteinGroup"].map(_accession)
    raw = raw.loc[raw["accession"].astype(str).str.len() > 0]
    abundance = raw.groupby("accession", sort=False)[sample_cols].median()
    abundance = abundance.apply(pd.to_numeric, errors="coerce")
    positive = abundance > 0
    return np.log2(abundance.where(positive))


def load_m001(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(data_dir / M001_META, sep="\t")
    meta["mouse_id"] = meta["UMN"].astype(str)
    meta["intervention"] = meta["Diet"].map(DIET_MAP)
    if meta["intervention"].isna().any():
        unknown = sorted(meta.loc[meta["intervention"].isna(), "Diet"].astype(str).unique())
        raise ValueError(f"Unmapped LC-M001 diets: {unknown}")
    meta["sex"] = meta["Sex"].astype(str)
    meta["experiment"] = "LC-M001"
    meta["organ"] = ORGAN
    raw = _drop_empty_columns(pd.read_csv(data_dir / M001_MATRIX, sep="\t"))
    col_map: dict[int, str] = {}
    for col in raw.columns:
        match = M001_COL_RE.search(str(col))
        if match:
            col_map[int(match.group(1))] = col
    missing = [int(i) for i in meta["ISB_ID"] if int(i) not in col_map]
    if missing:
        raise ValueError(f"LC-M001 matrix missing ISB_ID columns: {missing[:5]}")
    meta["sample_id"] = meta["ISB_ID"].map(lambda i: col_map[int(i)])
    sample_cols = meta["sample_id"].tolist()
    values = _log2_protein_matrix(raw, sample_cols)
    samples = meta[
        ["sample_id", "mouse_id", "ISB_ID", "experiment", "organ", "sex", "Diet", "intervention", "Liver (mg/ml)"]
    ].copy()
    samples = samples.rename(columns={"Liver (mg/ml)": "liver_mg_ml", "Diet": "diet_original"})
    return values, samples.reset_index(drop=True)


def load_m004(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(data_dir / M004_META, sep="\t")
    meta["mouse_id"] = meta["Mouse_ID"].astype(str)
    meta["sample_key"] = meta["mouse_id"]
    meta["intervention"] = meta["Treatment"].map(M004_TREATMENT_MAP)
    if meta["intervention"].isna().any():
        unknown = sorted(meta.loc[meta["intervention"].isna(), "Treatment"].astype(str).unique())
        raise ValueError(f"Unmapped LC-M004 treatments: {unknown}")
    meta["sex"] = meta["Sex"].astype(str)
    meta["experiment"] = "LC-M004"
    meta["organ"] = ORGAN
    raw = _drop_empty_columns(pd.read_csv(data_dir / M004_MATRIX, sep="\t"))
    col_map: dict[str, str] = {}
    for col in raw.columns:
        key = str(col).split(",")[0].strip()
        if key.startswith("T-"):
            col_map[key] = col
    missing = [k for k in meta["sample_key"] if k not in col_map]
    if missing:
        raise ValueError(f"LC-M004 matrix missing Mouse_ID columns: {missing[:5]}")
    meta["sample_id"] = meta["sample_key"].map(col_map)
    sample_cols = meta["sample_id"].tolist()
    values = _log2_protein_matrix(raw, sample_cols)
    samples = meta[
        ["sample_id", "mouse_id", "experiment", "organ", "sex", "Treatment", "intervention", "MS_File"]
    ].copy()
    samples = samples.rename(columns={"Treatment": "diet_original", "MS_File": "ms_file"})
    return values, samples.reset_index(drop=True)


def _score_subset(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    weights: pd.Series,
    clock_name: str,
    arms: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = [acc for acc in weights.index.astype(str) if acc in values.index]
    keep = [c for c in samples.columns if c in {
        "sample_id", "mouse_id", "ISB_ID", "experiment", "organ", "sex",
        "intervention", "diet_original", "liver_mg_ml", "ms_file",
    }]
    base = samples[keep].copy()
    base["clock"] = clock_name
    if len(overlap) < 3:
        base["score"] = np.nan
        base["n_proteins"] = len(overlap)
        metrics = pd.DataFrame(
            [
                {
                    "experiment": samples["experiment"].iloc[0],
                    "organ": ORGAN,
                    "sex": samples["sex"].iloc[0] if samples["sex"].nunique() == 1 else "pooled",
                    "clock": clock_name,
                    "intervention": arm,
                    "reference": "control",
                    "n_clock": int(len(weights)),
                    "n_overlap": len(overlap),
                    "overlap_frac": len(overlap) / len(weights) if len(weights) else np.nan,
                    "n_group": np.nan,
                    "n_reference": np.nan,
                    "median_group": np.nan,
                    "median_reference": np.nan,
                    "delta_median": np.nan,
                    "cliffs_delta": np.nan,
                    "mw_u": np.nan,
                    "p_parametric": np.nan,
                    "p_permutation": np.nan,
                    "note": "fewer_than_3_shared_proteins",
                }
                for arm in arms
                if arm != "control"
            ]
        )
        return base, metrics
    subset_cols = [c for c in samples["sample_id"].tolist() if c in values.columns]
    z = _zscore(values.loc[overlap, subset_cols])
    w = weights.reindex(z.index).astype(float)
    w = w[np.isfinite(w)]
    z = z.reindex(w.index)
    scored = z.T @ w
    base["score"] = scored.reindex(base["sample_id"]).to_numpy()
    base["n_proteins"] = int(len(w))
    sex_label = samples["sex"].iloc[0] if samples["sex"].nunique() == 1 else "pooled"
    metric_rows = []
    for arm in arms:
        if arm == "control":
            continue
        stats = _group_contrast(base, arm, reference="control")
        metric_rows.append(
            {
                "experiment": samples["experiment"].iloc[0],
                "organ": ORGAN,
                "sex": sex_label,
                "clock": clock_name,
                "intervention": arm,
                "reference": "control",
                "n_clock": int(len(weights)),
                "n_overlap": int(len(w)),
                "overlap_frac": len(w) / len(weights) if len(weights) else np.nan,
                **stats,
                "note": "",
            }
        )
    return base, pd.DataFrame(metric_rows)


def _run_experiment(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    weights: dict[str, pd.Series],
    arms: tuple[str, ...],
    sex: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    subset = samples.loc[samples["sex"] == sex].copy()
    score_frames = []
    metric_frames = []
    overlap_frames = []
    print(f"  {subset['experiment'].iloc[0]} {sex} n={len(subset)} ...", flush=True)
    abundance_ids = set(values.index.astype(str))
    for clock_name, w in weights.items():
        scores, metrics = _score_subset(values, subset, w, clock_name, arms)
        score_frames.append(scores)
        metric_frames.append(metrics)
        shared = sorted(set(w.index.astype(str)) & abundance_ids)
        overlap_frames.append(
            pd.DataFrame(
                {
                    "experiment": subset["experiment"].iloc[0],
                    "organ": ORGAN,
                    "clock": clock_name,
                    "accession": shared,
                    "weight": [float(w.loc[a]) for a in shared],
                }
            )
        )
        for _, hit in metrics.iterrows():
            print(
                f"    {clock_name} {hit['intervention']}: "
                f"{int(hit['n_overlap'])}/{int(hit['n_clock'])} proteins, "
                f"Δmed={hit['delta_median'] if pd.notna(hit['delta_median']) else float('nan'):.3f}, "
                f"δ={hit['cliffs_delta'] if pd.notna(hit['cliffs_delta']) else float('nan'):.3f} "
                f"(perm p={hit['p_permutation'] if pd.notna(hit['p_permutation']) else float('nan'):.3g})",
                flush=True,
            )
    return score_frames, metric_frames, overlap_frames


def write_orwell_transfer(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "orwell_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    weights = _frozen_weights(stab_dir, coef_dir, ORGAN)
    score_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    cohort_frames: list[pd.DataFrame] = []

    m001_values, m001_samples = load_m001(data_dir)
    m004_values, m004_samples = load_m004(data_dir)
    cohort_frames.extend([m001_samples, m004_samples])

    for sex in ("M", "F"):
        s, m, o = _run_experiment(m001_values, m001_samples, weights, M001_ARMS, sex)
        score_frames.extend(s)
        metric_frames.extend(m)
        if sex == "M":
            overlap_frames.extend(o)
        s, m, o = _run_experiment(m004_values, m004_samples, weights, M004_ARMS, sex)
        score_frames.extend(s)
        metric_frames.extend(m)
        if sex == "M":
            overlap_frames.extend(o)

    scores = pd.concat(score_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    cohort = pd.concat(cohort_frames, ignore_index=True)
    cohort.to_csv(dest / "cohort.csv", index=False)
    scores.to_csv(dest / "scores.csv", index=False)
    metrics.to_csv(dest / "metrics.csv", index=False)
    overlap.to_csv(dest / "overlap_proteins.csv", index=False)
    return {"scores": scores, "metrics": metrics, "overlap": overlap, "cohort": cohort}
