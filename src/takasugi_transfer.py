"""Apply frozen Scifo clocks to Takasugi whole-tissue TMT (no refit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ageaccel import permutation_spearman
from .shared_program import _pearson

TRANSFER_ORGANS = ["kidney", "liver", "lung", "heart", "brain", "muscle"]
STABLE_CUTOFF = 0.5
AGE_MAP = {"t06mo": 6.0, "t15mo": 15.0, "t24mo": 24.0, "t30mo": 30.0}

# Scifo organ -> Takasugi metadata sample + protein table (whole-tissue only).
WHOLE_TISSUE = {
    "heart": ("Heart_tissue", "Heart_whole_merged_results_20220531_0831_proteins.txt"),
    "kidney": ("Kidney_tissue", "Kidney_merged_results_20220530_2359_proteins.txt"),
    "lung": ("Lung_tissue", "Lung_merged_results_20220531_1121_proteins.txt"),
    "muscle": ("Muscle_tissue", "Muscle_merged_results_20220531_1130_proteins.txt"),
    "brain": ("brain", "brain_merged_results_20221213_2131_proteins.txt"),
    "liver": ("liver", "liver_merged_results_20221213_2127_proteins.txt"),
}


def _primary_accession(protein_id: str) -> str:
    token = str(protein_id).split("|")[0].strip()
    return token.split("-")[0] if token else ""


def _rel_fc_column(signal_col: str) -> str:
    suffix = str(signal_col).replace("signal_sum_", "")
    return f"rel_fc_{suffix}"


def load_takasugi_metadata(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("metadata_*.csv")):
        frame = pd.read_csv(path)
        frame["metadata_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No metadata_*.csv under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def load_whole_tissue(data_dir: Path, organ: str, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_name, filename = WHOLE_TISSUE[organ]
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    meta = metadata.loc[metadata["file"] == filename].copy()
    if "sample" in meta.columns:
        meta = meta.loc[meta["sample"] == sample_name]
    if meta.empty:
        raise ValueError(f"No metadata rows for {organ} ({filename}, {sample_name})")
    proteins = pd.read_csv(path, sep="\t")
    proteins["accession"] = proteins["protein_id"].map(_primary_accession)
    if "protein_fdr" in proteins.columns:
        proteins = proteins.loc[proteins["protein_fdr"].fillna(1) < 0.05]
    proteins = proteins.loc[proteins["accession"].astype(str).str.len() > 0]
    sample_rows = []
    for _, row in meta.iterrows():
        rel = _rel_fc_column(str(row["col.name"]))
        if rel not in proteins.columns:
            continue
        mouse_id = row["conditions_orig"] if "conditions_orig" in row.index else np.nan
        if pd.isna(mouse_id) or str(mouse_id).strip() == "":
            mouse_id = np.nan
        else:
            mouse_id = str(mouse_id).strip()
        sample_rows.append(
            {
                "sample_id": str(row["ID"]),
                "organ": organ,
                "age_months": AGE_MAP.get(str(row["condition"]), np.nan),
                "replicate": str(row["rep"]),
                "mouse_id": mouse_id,
                "rel_col": rel,
            }
        )
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        raise ValueError(f"No overlapping TMT channels for {organ}")
    values = proteins.set_index("accession")[samples["rel_col"].tolist()]
    values = values.apply(pd.to_numeric, errors="coerce")
    values.columns = samples["sample_id"].tolist()
    values = values.groupby(values.index).median()
    desc = proteins.drop_duplicates("accession").set_index("accession")
    if "description" in desc.columns:
        values.insert(0, "description", desc["description"].reindex(values.index))
    return values, samples


def _frozen_weights(stab_dir: Path, coef_dir: Path, organ: str) -> dict[str, pd.Series]:
    stab = pd.read_csv(stab_dir / f"{organ}.csv")
    stab["accession"] = stab["accession"].astype(str)
    stable = stab.loc[stab["selection_frequency"] > STABLE_CUTOFF].copy()
    weights = {
        "stable": stable.drop_duplicates("accession").set_index("accession")["mean_coef"],
    }
    coef_path = coef_dir / f"{organ}.csv"
    if coef_path.exists():
        coefs = pd.read_csv(coef_path)
        coefs["accession"] = coefs["accession"].astype(str)
        weights["full_fit"] = coefs.drop_duplicates("accession").set_index("accession")["coefficient"]
    return weights


def _zscore(block: pd.DataFrame) -> pd.DataFrame:
    filled = block.apply(lambda col: col.fillna(col.median()), axis=0)
    mu = filled.mean(axis=1)
    sd = filled.std(axis=1, ddof=0)
    keep = sd > 1e-12
    z = filled.loc[keep].sub(mu[keep], axis=0).div(sd[keep], axis=0)
    return z


def score_organ(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    weights: pd.Series,
    clock_name: str,
    n_clock: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abundance = values.drop(columns=["description"], errors="ignore")
    overlap = [acc for acc in weights.index.astype(str) if acc in abundance.index]
    n_overlap = len(overlap)
    if n_overlap < 3:
        keep_cols = [c for c in ["sample_id", "organ", "age_months", "replicate", "mouse_id"] if c in samples.columns]
        empty_scores = samples[keep_cols].copy()
        empty_scores["clock"] = clock_name
        empty_scores["score"] = np.nan
        empty_scores["n_proteins"] = n_overlap
        metrics = pd.DataFrame(
            [
                {
                    "organ": samples["organ"].iloc[0],
                    "clock": clock_name,
                    "n_clock": n_clock,
                    "n_overlap": n_overlap,
                    "overlap_frac": n_overlap / n_clock if n_clock else np.nan,
                    "n_samples": int(len(samples)),
                    "spearman_vs_age": np.nan,
                    "p_parametric": np.nan,
                    "p_permutation": np.nan,
                    "pearson_vs_age": np.nan,
                    "note": "fewer_than_3_shared_proteins",
                }
            ]
        )
        return empty_scores, metrics
    z = _zscore(abundance.loc[overlap])
    w = weights.reindex(z.index).astype(float)
    w = w[np.isfinite(w)]
    z = z.reindex(w.index)
    score = z.T @ w
    keep_cols = [c for c in ["sample_id", "organ", "age_months", "replicate", "mouse_id"] if c in samples.columns]
    out = samples[keep_cols].copy()
    out["clock"] = clock_name
    out["score"] = score.reindex(out["sample_id"]).to_numpy()
    out["n_proteins"] = int(len(w))
    ages = out["age_months"].to_numpy(dtype=float)
    vals = out["score"].to_numpy(dtype=float)
    r, p_param, p_perm = permutation_spearman(vals, ages)
    metrics = pd.DataFrame(
        [
            {
                "organ": out["organ"].iloc[0],
                "clock": clock_name,
                "n_clock": n_clock,
                "n_overlap": int(len(w)),
                "overlap_frac": len(w) / n_clock if n_clock else np.nan,
                "n_samples": int(len(out)),
                "spearman_vs_age": r,
                "p_parametric": p_param,
                "p_permutation": p_perm,
                "pearson_vs_age": _pearson(vals, ages),
                "note": "",
            }
        ]
    )
    return out, metrics


def write_takasugi_transfer(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "takasugi_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    metadata = load_takasugi_metadata(data_dir)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    score_frames = []
    metric_frames = []
    overlap_frames = []
    for organ in TRANSFER_ORGANS:
        print(f"  transfer {organ} ...", flush=True)
        values, samples = load_whole_tissue(data_dir, organ, metadata)
        weights = _frozen_weights(stab_dir, coef_dir, organ)
        abundance_ids = set(values.drop(columns=["description"], errors="ignore").index.astype(str))
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
                f"ρ={hit['spearman_vs_age'] if pd.notna(hit['spearman_vs_age']) else float('nan'):.3f} "
                f"(perm p={hit['p_permutation'] if pd.notna(hit['p_permutation']) else float('nan'):.3g})",
                flush=True,
            )
    scores = pd.concat(score_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    scores.to_csv(dest / "scores.csv", index=False)
    metrics.to_csv(dest / "metrics.csv", index=False)
    overlap.to_csv(dest / "overlap_proteins.csv", index=False)
    return {"scores": scores, "metrics": metrics, "overlap": overlap}
