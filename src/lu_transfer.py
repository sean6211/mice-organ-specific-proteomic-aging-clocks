"""Apply frozen Scifo clocks to Lu / Guo Cell 2025 mouse PulseDIA (no refit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ageaccel import permutation_spearman
from .burns_transfer import _group_contrast
from .shared_program import _pearson
from .takasugi_transfer import _frozen_weights, _zscore

MATRIX_NAME = "Aging_mouse_organ_PulseDIA_Matrix.tsv.xz"
META_NAME = "Aging_mouse_organ_SampleInformation.xlsx"
DIET_MAP = {"L": "lp", "M": "mp", "H": "hp"}
DIET_ARMS = ("lp", "mp", "hp")
# Scifo organ -> Lu Organ label (one tissue; do not pool brain regions).
ORGAN_MAP = {
    "brain": "Hippocampus",
    "heart": "Heart",
    "lung": "Lung",
    "liver": "Liver",
    "kidney": "Kidney",
    "spleen": "Spleen",
    "muscle": "Gastrocnemius muscle",
    "testis": "Testis",
}
TRANSFER_ORGANS = list(ORGAN_MAP)
GROUP_LABEL = {
    "G1": "young_male",
    "G4": "midlife_male",
    "G2": "old_male",
    "G3": "midlife_female",
}
# Primary PR contrast window in the paper.
PRIMARY_GROUP = "G4"
PRIMARY_SEX = "Male"


def _accession(token: str) -> str:
    return str(token).split("_")[0].strip()


def load_metadata(data_dir: Path) -> pd.DataFrame:
    path = data_dir / META_NAME
    meta = pd.read_excel(path)
    meta["sample_id"] = meta["Sample_name"].astype(str)
    meta["diet"] = meta["Diet"].map(DIET_MAP)
    meta["age_months"] = pd.to_numeric(meta["Month"], errors="coerce")
    meta["lu_organ"] = meta["Organ"].astype(str)
    meta["group_label"] = meta["Group"].map(GROUP_LABEL).fillna(meta["Group"].astype(str))
    return meta.reset_index(drop=True)


def load_matrix(data_dir: Path) -> pd.DataFrame:
    path = data_dir / MATRIX_NAME
    raw = pd.read_csv(path, sep="\t", index_col=0)
    raw.index = raw.index.map(_accession)
    values = raw.apply(pd.to_numeric, errors="coerce")
    values = values.groupby(values.index).median()
    values.columns = values.columns.map(str)
    return values


def _score_block(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    weights: pd.Series,
    clock_name: str,
) -> tuple[pd.DataFrame, int, int]:
    cols = [c for c in samples["sample_id"].tolist() if c in values.columns]
    overlap = [acc for acc in weights.index.astype(str) if acc in values.index]
    base = samples.copy()
    base["clock"] = clock_name
    if len(overlap) < 3 or len(cols) < 3:
        base["score"] = np.nan
        base["n_proteins"] = len(overlap)
        return base, len(weights), len(overlap)
    z = _zscore(values.loc[overlap, cols])
    w = weights.reindex(z.index).astype(float)
    w = w[np.isfinite(w)]
    z = z.reindex(w.index)
    scored = z.T @ w
    base["score"] = scored.reindex(base["sample_id"]).to_numpy()
    base["n_proteins"] = int(len(w))
    return base, int(len(weights)), int(len(w))


def write_lu_transfer(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "lu_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    organ_weights = {organ: _frozen_weights(stab_dir, coef_dir, organ) for organ in ORGAN_MAP}
    print("  loading Lu PulseDIA matrix ...", flush=True)
    values = load_matrix(data_dir)
    meta = load_metadata(data_dir)

    aging_scores = []
    aging_metrics = []
    diet_scores = []
    diet_metrics = []
    overlap_frames = []

    for organ, lu_name in ORGAN_MAP.items():
        weights = organ_weights[organ]
        aging = meta.loc[
            (meta["Type"] == "Aging")
            & (meta["Sex"] == "Male")
            & (meta["lu_organ"] == lu_name)
        ].copy()
        print(f"  {organ} ({lu_name}) aging n={len(aging)} ...", flush=True)
        for clock_name, w in weights.items():
            scored, n_clock, n_overlap = _score_block(values, aging, w, clock_name)
            scored["organ"] = organ
            scored["lu_organ"] = lu_name
            aging_scores.append(scored)
            ages = scored["age_months"].to_numpy(dtype=float)
            vals = scored["score"].to_numpy(dtype=float)
            rho, p_param, p_perm = permutation_spearman(vals, ages)
            aging_metrics.append(
                {
                    "organ": organ,
                    "lu_organ": lu_name,
                    "clock": clock_name,
                    "n_clock": n_clock,
                    "n_overlap": n_overlap,
                    "overlap_frac": n_overlap / n_clock if n_clock else np.nan,
                    "n_samples": int(np.isfinite(vals).sum()),
                    "spearman_vs_age": rho,
                    "p_parametric": p_param,
                    "p_permutation": p_perm,
                    "pearson_vs_age": _pearson(vals, ages),
                    "note": "z-scored within aging males of this organ; scores are not months",
                }
            )
            print(
                f"    aging {clock_name}: {n_overlap}/{n_clock}  ρ={rho if pd.notna(rho) else float('nan'):.3f}  "
                f"perm p={p_perm if pd.notna(p_perm) else float('nan'):.3g}",
                flush=True,
            )
            if clock_name == "stable":
                overlap_frames.append(
                    pd.DataFrame(
                        {
                            "organ": organ,
                            "lu_organ": lu_name,
                            "subset": "aging",
                            "accession": [a for a in w.index.astype(str) if a in values.index],
                        }
                    )
                )

        diet_all = meta.loc[(meta["Type"] == "Dietary_intervention") & (meta["lu_organ"] == lu_name)]
        for (group, sex), subset in diet_all.groupby(["Group", "Sex"], sort=False):
            subset = subset.copy()
            subset["intervention"] = subset["diet"]
            print(f"    diet {group} {sex} n={len(subset)} ...", flush=True)
            for clock_name, w in weights.items():
                scored, n_clock, n_overlap = _score_block(values, subset, w, clock_name)
                scored["organ"] = organ
                scored["lu_organ"] = lu_name
                diet_scores.append(scored)
                for arm, ref in (("lp", "mp"), ("hp", "mp"), ("lp", "hp")):
                    stats = _group_contrast(scored, arm, reference=ref)
                    diet_metrics.append(
                        {
                            "organ": organ,
                            "lu_organ": lu_name,
                            "clock": clock_name,
                            "group": group,
                            "group_label": GROUP_LABEL.get(str(group), str(group)),
                            "sex": sex,
                            "intervention": arm,
                            "reference": ref,
                            "n_clock": n_clock,
                            "n_overlap": n_overlap,
                            "overlap_frac": n_overlap / n_clock if n_clock else np.nan,
                            **stats,
                            "note": "z-scored within organ × start-age group × sex; scores are not months",
                        }
                    )
                    if clock_name == "stable":
                        print(
                            f"      {clock_name} {arm} vs {ref}: Δmed="
                            f"{stats['delta_median'] if pd.notna(stats['delta_median']) else float('nan'):.3f}  "
                            f"δ={stats['cliffs_delta'] if pd.notna(stats['cliffs_delta']) else float('nan'):.3f}  "
                            f"perm p={stats['p_permutation'] if pd.notna(stats['p_permutation']) else float('nan'):.3g}",
                            flush=True,
                        )

    aging_s = pd.concat(aging_scores, ignore_index=True)
    aging_m = pd.DataFrame(aging_metrics)
    diet_s = pd.concat(diet_scores, ignore_index=True)
    diet_m = pd.DataFrame(diet_metrics)
    overlap = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else pd.DataFrame()
    meta.to_csv(dest / "cohort.csv", index=False)
    aging_s.to_csv(dest / "aging_scores.csv", index=False)
    aging_m.to_csv(dest / "aging_metrics.csv", index=False)
    diet_s.to_csv(dest / "diet_scores.csv", index=False)
    diet_m.to_csv(dest / "diet_metrics.csv", index=False)
    overlap.to_csv(dest / "overlap_proteins.csv", index=False)
    return {
        "cohort": meta,
        "aging_scores": aging_s,
        "aging_metrics": aging_m,
        "diet_scores": diet_s,
        "diet_metrics": diet_m,
        "overlap": overlap,
    }
