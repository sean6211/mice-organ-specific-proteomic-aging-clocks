"""Apply frozen Scifo clocks to Burns UM-HET3 male young vs control (no refit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .ageaccel import permutation_mannwhitney, permutation_spearman
from .shared_program import _pearson
from .takasugi_transfer import _frozen_weights, _zscore

STABLE_CUTOFF = 0.5
BACKGROUND = "UM-HET3"
SEX = "M"
INTERVENTIONS = ("young", "control")
AGED_INTERVENTIONS = (
    "control",
    "acarbose",
    "canagliflozin",
    "rapamycin",
    "17aE2",
    "calorierestriction",
)
ORGAN_FILES = {
    "kidney": "lcproteomics_M005_peptide_proteotypic_unnormalized_kidney.csv",
    "liver": "lcproteomics_M005_peptide_proteotypic_unnormalized_liver.csv",
    "muscle": "lcproteomics_M005_peptide_proteotypic_unnormalized_gastroc.csv",
}
TRANSFER_ORGANS = list(ORGAN_FILES)


def _accession(protein: str) -> str:
    parts = str(protein).split("|")
    if len(parts) >= 2:
        return parts[1].split("-")[0].strip()
    return str(protein).split("-")[0].strip()


def load_metadata(
    data_dir: Path,
    interventions: tuple[str, ...] = INTERVENTIONS,
    *,
    background: str = BACKGROUND,
    sex: str = SEX,
) -> pd.DataFrame:
    path = data_dir / "lcproteomics_M005_sample_metadata.csv"
    meta = pd.read_csv(path)
    keep = (
        (meta["background"] == background)
        & (meta["sex"] == sex)
        & (meta["intervention"].isin(interventions))
    )
    meta = meta.loc[keep].copy()
    # Prefer the non-repeat aliquot when both exist for the same mouse + tissue.
    meta = meta.sort_values(["mouseid", "tissue", "repeat.", "sampleid"])
    meta = meta.drop_duplicates(["mouseid", "tissue"], keep="first")
    meta["organ"] = meta["tissue"].replace({"gastroc": "muscle"})
    meta["group"] = pd.Categorical(meta["intervention"], list(interventions), ordered=True)
    meta["age_rank"] = np.where(meta["intervention"] == "young", 0.0, 1.0)
    return meta.reset_index(drop=True)


def load_protein_matrix(data_dir: Path, organ: str, samples: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / ORGAN_FILES[organ]
    sample_ids = samples["sampleid"].astype(str).tolist()
    header = pd.read_csv(path, nrows=0)
    id_col = "Peptide" if "Peptide" in header.columns else "Peptides"
    missing = [s for s in sample_ids if s not in header.columns]
    if missing:
        raise ValueError(f"{path.name} missing {len(missing)} sample columns, e.g. {missing[:3]}")
    raw = pd.read_csv(path, usecols=[id_col, "Protein", *sample_ids])
    raw["accession"] = raw["Protein"].map(_accession)
    raw = raw.loc[raw["accession"].astype(str).str.len() > 0]
    abundance = raw.groupby("accession", sort=False)[sample_ids].median()
    abundance = abundance.apply(pd.to_numeric, errors="coerce")
    positive = abundance > 0
    logged = np.log2(abundance.where(positive))
    return logged


def _group_contrast(scores: pd.DataFrame, group: str, reference: str = "control") -> dict[str, float]:
    a = scores.loc[scores["intervention"] == group, "score"].to_numpy(dtype=float)
    b = scores.loc[scores["intervention"] == reference, "score"].to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    mw_u, p_param, p_perm = permutation_mannwhitney(a, b)
    if len(a) and len(b):
        u_stat = mannwhitneyu(a, b, alternative="two-sided").statistic
        cliffs = (2.0 * u_stat) / (len(a) * len(b)) - 1.0
        delta = float(np.median(a) - np.median(b))
    else:
        cliffs = np.nan
        delta = np.nan
    return {
        "n_group": int(len(a)),
        "n_reference": int(len(b)),
        "median_group": float(np.median(a)) if len(a) else np.nan,
        "median_reference": float(np.median(b)) if len(b) else np.nan,
        "delta_median": delta,
        "cliffs_delta": float(cliffs) if np.isfinite(cliffs) else np.nan,
        "mw_u": mw_u,
        "p_parametric": p_param,
        "p_permutation": p_perm,
    }


def score_organ(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    weights: pd.Series,
    clock_name: str,
    n_clock: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = [acc for acc in weights.index.astype(str) if acc in values.index]
    base = samples[["sampleid", "mouseid", "organ", "intervention", "group", "age_rank"]].copy()
    base = base.rename(columns={"sampleid": "sample_id", "mouseid": "mouse_id"})
    if len(overlap) < 3:
        empty = base.copy()
        empty["clock"] = clock_name
        empty["score"] = np.nan
        empty["n_proteins"] = len(overlap)
        metrics = pd.DataFrame(
            [
                {
                    "organ": samples["organ"].iloc[0],
                    "clock": clock_name,
                    "n_clock": n_clock,
                    "n_overlap": len(overlap),
                    "overlap_frac": len(overlap) / n_clock if n_clock else np.nan,
                    "n_young": int((samples["intervention"] == "young").sum()),
                    "n_control": int((samples["intervention"] == "control").sum()),
                    "median_young": np.nan,
                    "median_control": np.nan,
                    "delta_median": np.nan,
                    "cliffs_delta": np.nan,
                    "mw_u": np.nan,
                    "p_parametric": np.nan,
                    "p_permutation": np.nan,
                    "spearman_vs_group": np.nan,
                    "spearman_p_parametric": np.nan,
                    "spearman_p_permutation": np.nan,
                    "pearson_vs_group": np.nan,
                    "note": "fewer_than_3_shared_proteins",
                }
            ]
        )
        return empty, metrics
    z = _zscore(values.loc[overlap])
    w = weights.reindex(z.index).astype(float)
    w = w[np.isfinite(w)]
    z = z.reindex(w.index)
    score = z.T @ w
    out = base.copy()
    out["clock"] = clock_name
    out["score"] = score.reindex(out["sample_id"]).to_numpy()
    out["n_proteins"] = int(len(w))
    young = out.loc[out["intervention"] == "young", "score"].to_numpy(dtype=float)
    control = out.loc[out["intervention"] == "control", "score"].to_numpy(dtype=float)
    young = young[np.isfinite(young)]
    control = control[np.isfinite(control)]
    mw_u, p_param, p_perm = permutation_mannwhitney(control, young)
    if len(young) and len(control):
        u_stat = mannwhitneyu(control, young, alternative="two-sided").statistic
        cliffs = (2.0 * u_stat) / (len(control) * len(young)) - 1.0
    else:
        cliffs = np.nan
    r, r_p, r_perm = permutation_spearman(out["score"].to_numpy(dtype=float), out["age_rank"].to_numpy(dtype=float))
    metrics = pd.DataFrame(
        [
            {
                "organ": out["organ"].iloc[0],
                "clock": clock_name,
                "n_clock": n_clock,
                "n_overlap": int(len(w)),
                "overlap_frac": len(w) / n_clock if n_clock else np.nan,
                "n_young": int(len(young)),
                "n_control": int(len(control)),
                "median_young": float(np.median(young)) if len(young) else np.nan,
                "median_control": float(np.median(control)) if len(control) else np.nan,
                "delta_median": (
                    float(np.median(control) - np.median(young)) if len(young) and len(control) else np.nan
                ),
                "cliffs_delta": float(cliffs) if np.isfinite(cliffs) else np.nan,
                "mw_u": mw_u,
                "p_parametric": p_param,
                "p_permutation": p_perm,
                "spearman_vs_group": r,
                "spearman_p_parametric": r_p,
                "spearman_p_permutation": r_perm,
                "pearson_vs_group": _pearson(out["score"].to_numpy(dtype=float), out["age_rank"].to_numpy(dtype=float)),
                "note": "",
            }
        ]
    )
    return out, metrics


def write_burns_transfer(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "burns_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    score_frames = []
    metric_frames = []
    overlap_frames = []
    for organ in TRANSFER_ORGANS:
        samples = meta.loc[meta["organ"] == organ].copy()
        print(f"  transfer {organ} (n={len(samples)}) ...", flush=True)
        values = load_protein_matrix(data_dir, organ, samples)
        weights = _frozen_weights(stab_dir, coef_dir, organ)
        abundance_ids = set(values.index.astype(str))
        for clock_name, w in weights.items():
            scores, metrics = score_organ(values, samples, w, clock_name, n_clock=int(len(w)))
            score_frames.append(scores)
            metric_frames.append(metrics)
            shared = sorted(set(w.index.astype(str)) & abundance_ids)
            overlap_frames.append(
                pd.DataFrame(
                    {
                        "organ": organ,
                        "clock": clock_name,
                        "accession": shared,
                        "weight": [float(w.loc[a]) for a in shared],
                    }
                )
            )
            hit = metrics.iloc[0]
            print(
                f"    {clock_name}: {int(hit['n_overlap'])}/{int(hit['n_clock'])} proteins, "
                f"Δmedian={hit['delta_median'] if pd.notna(hit['delta_median']) else float('nan'):.3f}, "
                f"Cliff δ={hit['cliffs_delta'] if pd.notna(hit['cliffs_delta']) else float('nan'):.3f} "
                f"(perm p={hit['p_permutation'] if pd.notna(hit['p_permutation']) else float('nan'):.3g})",
                flush=True,
            )
    scores = pd.concat(score_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    meta.to_csv(dest / "cohort.csv", index=False)
    scores.to_csv(dest / "scores.csv", index=False)
    metrics.to_csv(dest / "metrics.csv", index=False)
    overlap.to_csv(dest / "overlap_proteins.csv", index=False)
    interventions = write_burns_interventions(data_dir, out_dir)
    return {
        "scores": scores,
        "metrics": metrics,
        "overlap": overlap,
        "cohort": meta,
        **{f"intervention_{k}": v for k, v in interventions.items()},
    }


def write_burns_interventions(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Frozen scores for aged UM-HET3 males; each ITP arm vs aged control."""
    dest = out_dir / "burns_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir, interventions=AGED_INTERVENTIONS)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    score_frames = []
    metric_frames = []
    print("  aged UM-HET3 male interventions vs control ...", flush=True)
    for organ in TRANSFER_ORGANS:
        samples = meta.loc[meta["organ"] == organ].copy()
        print(f"    {organ} n={len(samples)}", flush=True)
        values = load_protein_matrix(data_dir, organ, samples)
        weights = _frozen_weights(stab_dir, coef_dir, organ)
        for clock_name, w in weights.items():
            overlap = [acc for acc in w.index.astype(str) if acc in values.index]
            base = samples[["sampleid", "mouseid", "organ", "intervention", "group"]].copy()
            base = base.rename(columns={"sampleid": "sample_id", "mouseid": "mouse_id"})
            base["clock"] = clock_name
            if len(overlap) < 3:
                base["score"] = np.nan
                base["n_proteins"] = len(overlap)
                score_frames.append(base)
                continue
            z = _zscore(values.loc[overlap])
            ww = w.reindex(z.index).astype(float)
            ww = ww[np.isfinite(ww)]
            z = z.reindex(ww.index)
            scored = z.T @ ww
            base["score"] = scored.reindex(base["sample_id"]).to_numpy()
            base["n_proteins"] = int(len(ww))
            score_frames.append(base)
            for arm in AGED_INTERVENTIONS:
                if arm == "control":
                    continue
                stats = _group_contrast(base, arm, reference="control")
                metric_frames.append(
                    {
                        "organ": organ,
                        "clock": clock_name,
                        "intervention": arm,
                        "reference": "control",
                        "n_clock": int(len(w)),
                        "n_overlap": int(len(ww)),
                        "overlap_frac": len(ww) / len(w) if len(w) else np.nan,
                        **stats,
                    }
                )
                print(
                    f"      {clock_name} {arm}: Δmed={stats['delta_median']:.3f} "
                    f"δ={stats['cliffs_delta']:.3f} perm p={stats['p_permutation']:.3g}",
                    flush=True,
                )
    scores = pd.concat(score_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_frames)
    meta.to_csv(dest / "intervention_cohort.csv", index=False)
    scores.to_csv(dest / "intervention_scores.csv", index=False)
    metrics.to_csv(dest / "intervention_metrics.csv", index=False)
    return {"scores": scores, "metrics": metrics, "cohort": meta}
