"""Paired Takasugi RNA vs TMT layer test of frozen Scifo clocks. No refit.

Within-tissue join only: RNA {age}M{k} ↔ TMT t{age}mo / rep{k}.
Matched genes = stable clock accessions with TMT gene_name present in RNA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

from .ageaccel import N_PERM, ageaccel_l, permutation_spearman
from .steiger import steiger_z
from .shared_program import _pearson
from .takasugi_transfer import (
    TRANSFER_ORGANS,
    WHOLE_TISSUE,
    _frozen_weights,
    _primary_accession,
    _zscore,
    load_takasugi_metadata,
    load_whole_tissue,
)

RNA_DIRNAME = "GSE225576"
RNA_FILES = {
    "brain": "GSE225576_Brain.txt.gz",
    "heart": "GSE225576_Heart.txt.gz",
    "kidney": "GSE225576_Kidney.txt.gz",
    "liver": "GSE225576_Liver.txt.gz",
    "lung": "GSE225576_Lung.txt.gz",
    "muscle": "GSE225576_Muscle.txt.gz",
}
RHO_CUT = 0.2
N_NULL = 1_000
MIN_MATCHED = 3
RNG = np.random.default_rng(0)


def _normalize_gene(name: object) -> str:
    if name is None or (isinstance(name, float) and not np.isfinite(name)):
        return ""
    token = str(name).strip()
    if token in {"", "nan", "None", "NA"}:
        return ""
    return token.split(";")[0].split(",")[0].strip()


def rna_column(age_months: float, replicate: str) -> str:
    k = int(str(replicate).lower().replace("rep", ""))
    return f"{int(float(age_months))}M{k}"


def load_rna_matrix(data_dir: Path, organ: str) -> pd.DataFrame:
    path = data_dir / RNA_DIRNAME / RNA_FILES[organ]
    if not path.exists():
        raise FileNotFoundError(path)
    rna = pd.read_csv(path, sep="\t", compression="gzip")
    if "Gene_ID" not in rna.columns:
        raise ValueError(f"No Gene_ID column in {path}")
    rna["Gene_ID"] = rna["Gene_ID"].map(_normalize_gene)
    rna = rna.loc[rna["Gene_ID"].astype(str).str.len() > 0]
    numeric = rna.drop(columns=["Gene_ID"]).apply(pd.to_numeric, errors="coerce")
    numeric.index = rna["Gene_ID"].to_numpy()
    numeric = numeric.groupby(numeric.index).median()
    return numeric.clip(lower=0.0)


def load_accession_genes(data_dir: Path, organ: str, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    sample_name, filename = WHOLE_TISSUE[organ]
    proteins = pd.read_csv(data_dir / filename, sep="\t")
    proteins["accession"] = proteins["protein_id"].map(_primary_accession)
    if "protein_fdr" in proteins.columns:
        proteins = proteins.loc[proteins["protein_fdr"].fillna(1) < 0.05]
    proteins = proteins.loc[proteins["accession"].astype(str).str.len() > 0]
    if "gene_name" not in proteins.columns:
        proteins["gene_name"] = ""
    proteins["gene"] = proteins["gene_name"].map(_normalize_gene)
    out = proteins.loc[proteins["gene"].astype(str).str.len() > 0, ["accession", "gene"]].drop_duplicates()
    out = out.drop_duplicates("accession", keep="first")
    out["organ"] = organ
    return out.reset_index(drop=True)


def _align_samples(samples: pd.DataFrame, rna: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    out["rna_col"] = [rna_column(a, r) for a, r in zip(out["age_months"], out["replicate"])]
    out = out.loc[out["rna_col"].isin(rna.columns)].copy()
    return out.reset_index(drop=True)


def _gene_weights_and_protein(
    abundance: pd.DataFrame,
    acc_genes: pd.DataFrame,
    weights: pd.Series,
    rna_genes: set[str],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Collapse clock accessions to genes present in both TMT and RNA."""
    gene_of = acc_genes.drop_duplicates("accession").set_index("accession")["gene"]
    rows = []
    protein_blocks: dict[str, list[str]] = {}
    for acc, weight in weights.items():
        acc = str(acc)
        gene = str(gene_of[acc]) if acc in gene_of.index else ""
        in_tmt = acc in abundance.index
        in_rna = gene in rna_genes
        rows.append(
            {
                "accession": acc,
                "gene": gene,
                "weight": float(weight),
                "in_tmt": bool(in_tmt),
                "in_rna": bool(in_rna),
                "matched": bool(in_tmt and in_rna and gene),
            }
        )
        if in_tmt and in_rna and gene:
            protein_blocks.setdefault(gene, []).append(acc)
    gene_map = pd.DataFrame(rows)
    if not protein_blocks:
        return pd.Series(dtype=float), pd.DataFrame(), gene_map
    prot_cols = {}
    w_gene = {}
    for gene, accs in protein_blocks.items():
        prot_cols[gene] = abundance.loc[accs].median(axis=0)
        w_gene[gene] = float(gene_map.loc[gene_map["gene"] == gene, "weight"].sum())
    protein = pd.DataFrame(prot_cols).T
    weights_gene = pd.Series(w_gene, dtype=float)
    weights_gene = weights_gene.reindex(protein.index)
    return weights_gene, protein, gene_map


def _score_matched(block: pd.DataFrame, weights: pd.Series) -> pd.Series:
    z = _zscore(block.reindex(weights.index))
    w = weights.reindex(z.index).astype(float)
    w = w[np.isfinite(w)]
    z = z.reindex(w.index)
    if len(w) < MIN_MATCHED:
        return pd.Series(np.nan, index=block.columns)
    return z.T @ w


def _spearman_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    xa = x[mask]
    ya = y[mask]
    if np.nanstd(xa) < 1e-12 or np.nanstd(ya) < 1e-12:
        return float("nan"), float("nan")
    r, p = spearmanr(xa, ya)
    return float(r), float(p)


def _layer_class(rho_protein_age: float, rho_rna_age: float) -> str:
    prot = np.isfinite(rho_protein_age) and abs(rho_protein_age) >= RHO_CUT
    rna = np.isfinite(rho_rna_age) and abs(rho_rna_age) >= RHO_CUT
    if prot and rna:
        return "both"
    if prot:
        return "protein_driven"
    if rna:
        return "rna_driven"
    return "neither"


def _background_protein_by_gene(
    abundance: pd.DataFrame,
    acc_genes: pd.DataFrame,
    clock_genes: set[str],
    rna_genes: set[str],
) -> pd.DataFrame:
    gene_of = acc_genes.drop_duplicates("accession").set_index("accession")["gene"]
    blocks: dict[str, list[str]] = {}
    for acc in abundance.index.astype(str):
        gene = str(gene_of[acc]) if acc in gene_of.index else ""
        if not gene or gene in clock_genes or gene not in rna_genes:
            continue
        blocks.setdefault(gene, []).append(acc)
    if not blocks:
        return pd.DataFrame()
    return pd.DataFrame({gene: abundance.loc[accs].median(axis=0) for gene, accs in blocks.items()}).T


def _rna_null_rhos(
    rna_log: pd.DataFrame,
    weights: pd.Series,
    ages: np.ndarray,
    n_null: int,
) -> np.ndarray:
    pool = rna_log.index.astype(str).tolist()
    k = int(len(weights))
    if k < MIN_MATCHED or len(pool) < k:
        return np.full(n_null, np.nan)
    w_vals = weights.to_numpy(dtype=float)
    rhos = np.empty(n_null, dtype=float)
    for i in range(n_null):
        genes = RNG.choice(pool, size=k, replace=False)
        w_rand = pd.Series(RNG.permutation(w_vals), index=genes)
        scores = _score_matched(rna_log.loc[list(genes)], w_rand)
        r, _ = _spearman_pair(scores.to_numpy(dtype=float), ages)
        rhos[i] = r
    return rhos


def _enrichment_protein_driven(clock_class: np.ndarray, bg_class: np.ndarray) -> dict[str, float]:
    clock_hit = int((clock_class == "protein_driven").sum())
    clock_n = int(len(clock_class))
    bg_hit = int((bg_class == "protein_driven").sum())
    bg_n = int(len(bg_class))
    table = np.array(
        [
            [clock_hit, clock_n - clock_hit],
            [bg_hit, bg_n - bg_hit],
        ],
        dtype=float,
    )
    if clock_n < 3 or bg_n < 3 or table.min() < 0:
        return {
            "clock_n": clock_n,
            "clock_protein_driven": clock_hit,
            "clock_frac": clock_hit / clock_n if clock_n else np.nan,
            "background_n": bg_n,
            "background_protein_driven": bg_hit,
            "background_frac": bg_hit / bg_n if bg_n else np.nan,
            "odds_ratio": np.nan,
            "p_fisher": np.nan,
            "p_permutation": np.nan,
        }
    odds, p_fish = fisher_exact(table, alternative="greater")
    labels = np.concatenate(
        [np.ones(clock_n, dtype=bool), np.zeros(bg_n, dtype=bool)]
    )
    driven = np.concatenate(
        [clock_class == "protein_driven", bg_class == "protein_driven"]
    )
    obs = clock_hit
    count = 0
    work = labels.copy()
    for _ in range(N_PERM):
        RNG.shuffle(work)
        if int(driven[work].sum()) >= obs:
            count += 1
    p_perm = (count + 1) / (N_PERM + 1)
    return {
        "clock_n": clock_n,
        "clock_protein_driven": clock_hit,
        "clock_frac": clock_hit / clock_n,
        "background_n": bg_n,
        "background_protein_driven": bg_hit,
        "background_frac": bg_hit / bg_n,
        "odds_ratio": float(odds),
        "p_fisher": float(p_fish),
        "p_permutation": float(p_perm),
    }


def analyze_organ(
    data_dir: Path,
    organ: str,
    metadata: pd.DataFrame,
    weights: pd.Series,
) -> dict[str, pd.DataFrame]:
    values, samples = load_whole_tissue(data_dir, organ, metadata)
    abundance = values.drop(columns=["description"], errors="ignore")
    acc_genes = load_accession_genes(data_dir, organ, metadata)
    rna = load_rna_matrix(data_dir, organ)
    samples = _align_samples(samples, rna)
    if samples.empty:
        raise ValueError(f"No RNA columns matched TMT samples for {organ}")
    rna_genes = set(rna.index.astype(str))
    w_gene, protein_gene, gene_map = _gene_weights_and_protein(
        abundance, acc_genes, weights, rna_genes
    )
    gene_map["organ"] = organ
    matched = [g for g in w_gene.index if g in rna_genes]
    n_clock = int(len(weights))
    n_matched = int(len(matched))
    if n_matched < MIN_MATCHED:
        empty_scores = samples[["sample_id", "organ", "age_months", "replicate"]].copy()
        if "mouse_id" in samples.columns:
            empty_scores["mouse_id"] = samples["mouse_id"]
        empty_scores["rna_col"] = samples["rna_col"]
        empty_scores["score_protein"] = np.nan
        empty_scores["score_rna"] = np.nan
        empty_scores["ageaccel_protein"] = np.nan
        empty_scores["ageaccel_rna"] = np.nan
        empty_scores["n_matched"] = n_matched
        metrics = pd.DataFrame(
            [
                {
                    "organ": organ,
                    "n_clock": n_clock,
                    "n_matched": n_matched,
                    "n_samples": int(len(samples)),
                    "note": "fewer_than_3_matched_genes",
                }
            ]
        )
        return {
            "gene_map": gene_map,
            "scores": empty_scores,
            "metrics": metrics,
            "gene_layer": pd.DataFrame(),
            "gene_layer_tests": pd.DataFrame(
                [{"organ": organ, "note": "fewer_than_3_matched_genes"}]
            ),
            "rna_null": pd.DataFrame([{"organ": organ, "n_null": 0, "note": "fewer_than_3_matched_genes"}]),
        }

    sample_ids = samples["sample_id"].tolist()
    rna_cols = samples["rna_col"].tolist()
    prot = protein_gene.reindex(index=matched, columns=sample_ids)
    rna_matched = rna.reindex(index=matched, columns=rna_cols)
    rna_matched.columns = sample_ids
    rna_log = np.log2(rna_matched + 1.0)
    w_use = w_gene.reindex(matched).astype(float)
    score_protein = _score_matched(prot, w_use)
    score_rna = _score_matched(rna_log, w_use)
    ages = samples["age_months"].to_numpy(dtype=float)
    prot_vals = score_protein.reindex(sample_ids).to_numpy(dtype=float)
    rna_vals = score_rna.reindex(sample_ids).to_numpy(dtype=float)
    aa_prot = ageaccel_l(prot_vals, ages)
    aa_rna = ageaccel_l(rna_vals, ages)

    scores = samples[["sample_id", "organ", "age_months", "replicate"]].copy()
    if "mouse_id" in samples.columns:
        scores["mouse_id"] = samples["mouse_id"]
    scores["rna_col"] = samples["rna_col"]
    scores["score_protein"] = prot_vals
    scores["score_rna"] = rna_vals
    scores["ageaccel_protein"] = aa_prot
    scores["ageaccel_rna"] = aa_rna
    scores["n_matched"] = n_matched

    r_p, p_p, pp_p = permutation_spearman(prot_vals, ages)
    r_r, p_r, pp_r = permutation_spearman(rna_vals, ages)
    r_pr, p_pr, pp_pr = permutation_spearman(rna_vals, prot_vals)
    r_aa, p_aa, pp_aa = permutation_spearman(aa_rna, aa_prot)
    z_st, p_st = steiger_z(r_p, r_r, r_pr, int(np.isfinite(prot_vals).sum()))

    rna_all_log = np.log2(rna.clip(lower=0.0) + 1.0)
    null_rhos = _rna_null_rhos(rna_all_log, w_use, ages, N_NULL)
    finite_null = null_rhos[np.isfinite(null_rhos)]
    if np.isfinite(r_r) and len(finite_null):
        n_ge = int(np.sum(finite_null >= r_r))
        p_null = (n_ge + 1) / (len(finite_null) + 1)
        percentile = float(100.0 * np.mean(finite_null < r_r))
        null_mean = float(np.mean(finite_null))
        null_q95 = float(np.quantile(finite_null, 0.95))
    else:
        p_null = null_mean = null_q95 = percentile = np.nan

    metrics = pd.DataFrame(
        [
            {
                "organ": organ,
                "n_clock": n_clock,
                "n_matched": n_matched,
                "overlap_frac": n_matched / n_clock if n_clock else np.nan,
                "n_samples": int(len(samples)),
                "spearman_protein_vs_age": r_p,
                "p_parametric_protein_vs_age": p_p,
                "p_permutation_protein_vs_age": pp_p,
                "pearson_protein_vs_age": _pearson(prot_vals, ages),
                "spearman_rna_vs_age": r_r,
                "p_parametric_rna_vs_age": p_r,
                "p_permutation_rna_vs_age": pp_r,
                "pearson_rna_vs_age": _pearson(rna_vals, ages),
                "spearman_rna_vs_protein": r_pr,
                "p_parametric_rna_vs_protein": p_pr,
                "p_permutation_rna_vs_protein": pp_pr,
                "pearson_rna_vs_protein": _pearson(rna_vals, prot_vals),
                "steiger_z": z_st,
                "steiger_p": p_st,
                "spearman_ageaccel_rna_vs_protein": r_aa,
                "p_parametric_ageaccel": p_aa,
                "p_permutation_ageaccel": pp_aa,
                "rna_null_n": int(len(finite_null)),
                "rna_null_mean": null_mean,
                "rna_null_q95": null_q95,
                "rna_null_p": p_null,
                "rna_null_percentile": percentile,
                "note": "",
            }
        ]
    )
    rna_null = pd.DataFrame(
        [
            {
                "organ": organ,
                "n_null": int(len(finite_null)),
                "observed_spearman_rna_vs_age": r_r,
                "null_mean": null_mean,
                "null_q95": null_q95,
                "p_null": p_null,
                "percentile": percentile,
            }
        ]
    )

    layer_rows = []
    ages_v = ages
    for gene in matched:
        p_vec = prot.loc[gene, sample_ids].to_numpy(dtype=float)
        r_vec = rna_log.loc[gene, sample_ids].to_numpy(dtype=float)
        rp, pp = _spearman_pair(p_vec, ages_v)
        rr, pr = _spearman_pair(r_vec, ages_v)
        rpr, ppr = _spearman_pair(r_vec, p_vec)
        layer_rows.append(
            {
                "organ": organ,
                "gene": gene,
                "set": "clock",
                "weight": float(w_use.loc[gene]),
                "spearman_protein_vs_age": rp,
                "p_protein_vs_age": pp,
                "spearman_rna_vs_age": rr,
                "p_rna_vs_age": pr,
                "spearman_rna_vs_protein": rpr,
                "p_rna_vs_protein": ppr,
                "layer_class": _layer_class(rp, rr),
            }
        )

    bg_prot = _background_protein_by_gene(abundance, acc_genes, set(matched), rna_genes)
    bg_genes = [g for g in bg_prot.index if g in rna_genes]
    rna_bg = np.log2(rna.reindex(index=bg_genes, columns=rna_cols).clip(lower=0.0) + 1.0)
    rna_bg.columns = sample_ids
    bg_prot = bg_prot.reindex(columns=sample_ids)
    for gene in bg_genes:
        p_vec = bg_prot.loc[gene, sample_ids].to_numpy(dtype=float)
        r_vec = rna_bg.loc[gene, sample_ids].to_numpy(dtype=float)
        rp, pp = _spearman_pair(p_vec, ages_v)
        rr, pr = _spearman_pair(r_vec, ages_v)
        rpr, ppr = _spearman_pair(r_vec, p_vec)
        layer_rows.append(
            {
                "organ": organ,
                "gene": gene,
                "set": "background",
                "weight": np.nan,
                "spearman_protein_vs_age": rp,
                "p_protein_vs_age": pp,
                "spearman_rna_vs_age": rr,
                "p_rna_vs_age": pr,
                "spearman_rna_vs_protein": rpr,
                "p_rna_vs_protein": ppr,
                "layer_class": _layer_class(rp, rr),
            }
        )
    gene_layer = pd.DataFrame(layer_rows)
    clock_class = gene_layer.loc[gene_layer["set"] == "clock", "layer_class"].to_numpy()
    bg_class = gene_layer.loc[gene_layer["set"] == "background", "layer_class"].to_numpy()
    tests = _enrichment_protein_driven(clock_class, bg_class)
    tests["organ"] = organ
    for cls_name in ("both", "protein_driven", "rna_driven", "neither"):
        tests[f"clock_{cls_name}"] = int((clock_class == cls_name).sum())
        tests[f"background_{cls_name}"] = int((bg_class == cls_name).sum())
    gene_layer_tests = pd.DataFrame([tests])
    return {
        "gene_map": gene_map,
        "scores": scores,
        "metrics": metrics,
        "gene_layer": gene_layer,
        "gene_layer_tests": gene_layer_tests,
        "rna_null": rna_null,
    }


def write_takasugi_rna(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "takasugi_rna"
    dest.mkdir(parents=True, exist_ok=True)
    metadata = load_takasugi_metadata(data_dir)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    buckets = {key: [] for key in ("gene_map", "scores", "metrics", "gene_layer", "gene_layer_tests", "rna_null")}
    for organ in TRANSFER_ORGANS:
        print(f"  RNA–protein layer {organ} ...", flush=True)
        weights = _frozen_weights(stab_dir, coef_dir, organ)["stable"]
        tables = analyze_organ(data_dir, organ, metadata, weights)
        for key, frame in tables.items():
            if not frame.empty:
                buckets[key].append(frame)
        hit = tables["metrics"].iloc[0]
        rp = hit.get("spearman_protein_vs_age", float("nan"))
        rr = hit.get("spearman_rna_vs_age", float("nan"))
        rp_s = f"{float(rp):.3f}" if pd.notna(rp) else "NA"
        rr_s = f"{float(rr):.3f}" if pd.notna(rr) else "NA"
        print(
            f"    matched {int(hit.get('n_matched', 0))}/{int(hit.get('n_clock', 0))} genes  "
            f"protein ρ={rp_s}  RNA ρ={rr_s}",
            flush=True,
        )
    out = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in buckets.items()
    }
    out["gene_map"].to_csv(dest / "gene_map.csv", index=False)
    out["scores"].to_csv(dest / "scores.csv", index=False)
    out["metrics"].to_csv(dest / "metrics.csv", index=False)
    out["gene_layer"].to_csv(dest / "gene_layer.csv", index=False)
    out["gene_layer_tests"].to_csv(dest / "gene_layer_tests.csv", index=False)
    out["rna_null"].to_csv(dest / "rna_null_draws.csv", index=False)
    return out
