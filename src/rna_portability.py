"""Hard nulls, score anatomy, and protein↔RNA age-slope concordance.

Frozen Scifo stable weights. No refit. Takasugi paired RNA/TMT is the primary
layer test; Lu aging males get the same gene/sign/weight decomposition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .ageaccel import permutation_spearman
from .paths import project_root
from .clock_explain import _load_processed
from .lu_transfer import ORGAN_MAP as LU_ORGAN_MAP
from .lu_transfer import load_matrix as load_lu_matrix
from .lu_transfer import load_metadata as load_lu_metadata
from .shared_program import _pearson
from .takasugi_rna import (
    MIN_MATCHED,
    TRANSFER_ORGANS,
    _align_samples,
    _gene_weights_and_protein,
    _score_matched,
    _spearman_pair,
    load_accession_genes,
    load_rna_matrix,
)
from .takasugi_transfer import (
    _frozen_weights,
    load_takasugi_metadata,
    load_whole_tissue,
)

N_NULL = 1_000
RHO_CUT = 0.2
RNG = np.random.default_rng(1)

NULL_ORDER = [
    "random_detected",
    "abundance_matched",
    "variance_matched",
    "missingness_matched",
    "age_rho_matched",
    "de_status_matched",
    "module_matched",
    "weight_permuted",
    "sign_permuted",
]

NULL_LABEL = {
    "random_detected": "Random detected genes",
    "abundance_matched": "Abundance-matched",
    "variance_matched": "Variance-matched",
    "missingness_matched": "Missingness-matched",
    "age_rho_matched": "Training age-ρ matched",
    "de_status_matched": "DE-status matched",
    "module_matched": "Module-matched",
    "weight_permuted": "Weight-permuted clock genes",
    "sign_permuted": "Sign-permuted clock genes",
}

SCORE_ORDER = ["exact_weight", "sign_only", "unweighted"]
SCORE_LABEL = {
    "exact_weight": "Exact weights",
    "sign_only": "Sign only",
    "unweighted": "Unweighted gene set",
}


def _spearman_rows(values: pd.DataFrame, ages: np.ndarray) -> pd.Series:
    ages = np.asarray(ages, dtype=float)
    rhos = []
    index = []
    for acc in values.index.astype(str):
        y = values.loc[acc].to_numpy(dtype=float)
        r, _ = _spearman_pair(y, ages)
        rhos.append(r)
        index.append(acc)
    return pd.Series(rhos, index=index, dtype=float)


def _row_stats(values: pd.DataFrame) -> pd.DataFrame:
    arr = values.to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=1)
        var = np.nanvar(arr, axis=1)
        obs = np.isfinite(arr).mean(axis=1)
    return pd.DataFrame(
        {"mean": mean, "variance": var, "observed_frac": obs},
        index=values.index.astype(str),
    )


def _collapse_gene(frame: pd.DataFrame, gene_of: pd.Series) -> pd.DataFrame:
    work = frame.copy()
    work["gene"] = work.index.astype(str).map(gene_of)
    work = work.loc[work["gene"].astype(str).str.len() > 0]
    if work.empty:
        return pd.DataFrame()
    rows = []
    for gene, sub in work.groupby("gene"):
        rows.append(
            {
                "gene": str(gene),
                "mean": float(sub["mean"].median()),
                "variance": float(sub["variance"].median()),
                "observed_frac": float(sub["observed_frac"].median()),
                "spearman_vs_age": float(sub["spearman_vs_age"].median())
                if "spearman_vs_age" in sub
                else np.nan,
                "age_de": bool(sub["age_de"].any()) if "age_de" in sub else False,
                "module": str(sub["module"].iloc[0]) if "module" in sub else "unassigned",
            }
        )
    return pd.DataFrame(rows).set_index("gene")


def _expand_unused_window(
    used: np.ndarray,
    values: np.ndarray,
    center: int,
    x: float,
    window: int,
) -> np.ndarray:
    n = int(len(values))
    left, right = center - 1, center
    cand: list[int] = []
    while len(cand) < window and (left >= 0 or right < n):
        take_left = False
        if left >= 0 and right < n:
            take_left = abs(float(values[left]) - x) <= abs(float(values[right]) - x)
        elif left >= 0:
            take_left = True
        if take_left:
            if not used[left]:
                cand.append(left)
            left -= 1
        else:
            if not used[right]:
                cand.append(right)
            right += 1
    return np.asarray(cand, dtype=int)


def _match_values(
    clock_values: np.ndarray,
    pool_ids: np.ndarray,
    pool_values: np.ndarray,
    rng: np.random.Generator,
    window: int = 40,
) -> list[str] | None:
    k = int(len(clock_values))
    mask = np.isfinite(pool_values)
    ids = np.asarray(pool_ids)[mask]
    vals = np.asarray(pool_values, dtype=float)[mask]
    if k < MIN_MATCHED or len(ids) < k:
        return None
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    ids = ids[order]
    used = np.zeros(len(ids), dtype=bool)
    chosen: list[str | None] = [None] * k
    for i in rng.permutation(k):
        avail = np.flatnonzero(~used)
        if avail.size == 0:
            return None
        x = clock_values[i]
        if np.isfinite(x):
            center = int(np.searchsorted(vals, x, side="left"))
            keep = _expand_unused_window(used, vals, center, float(x), window)
            pick = int(rng.choice(keep)) if keep.size else int(rng.choice(avail))
        else:
            pick = int(rng.choice(avail))
        chosen[i] = str(ids[pick])
        used[pick] = True
    if any(c is None for c in chosen):
        return None
    return [str(c) for c in chosen]


def _match_discrete(
    clock_labels: np.ndarray,
    pool_ids: np.ndarray,
    pool_labels: np.ndarray,
    rng: np.random.Generator,
) -> list[str] | None:
    remaining: dict[object, list] = {}
    for lab, idx in zip(pool_labels, pool_ids):
        remaining.setdefault(lab, []).append(idx)
    for lab in remaining:
        rng.shuffle(remaining[lab])
    chosen = []
    for lab in clock_labels:
        bucket = remaining.get(lab)
        if not bucket:
            return None
        chosen.append(str(bucket.pop()))
    return chosen


def _zscore_np(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        med = np.nanmedian(mat, axis=1, keepdims=True)
        filled = np.where(np.isfinite(mat), mat, med)
        mu = filled.mean(axis=1, keepdims=True)
        sd = filled.std(axis=1, ddof=0, keepdims=True)
    keep = np.isfinite(sd.ravel()) & (sd.ravel() > 1e-12)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (filled - mu) / sd, keep


def _score_rho_np(mat: np.ndarray, weights: np.ndarray, ages: np.ndarray) -> float:
    if mat.size == 0 or len(weights) < MIN_MATCHED:
        return float("nan")
    z, keep = _zscore_np(mat)
    if int(keep.sum()) < MIN_MATCHED:
        return float("nan")
    scores = z[keep].T @ weights[keep]
    r, _ = _spearman_pair(scores, ages)
    return r


def _score_rho(block: pd.DataFrame, weights: pd.Series, ages: np.ndarray) -> float:
    scores = _score_matched(block, weights)
    r, _ = _spearman_pair(scores.to_numpy(dtype=float), ages)
    return r


def _score_variants(block: pd.DataFrame, weights: pd.Series) -> dict[str, pd.Series]:
    w = weights.astype(float)
    w = w[np.isfinite(w)]
    z_block = block.reindex(w.index)
    exact = _score_matched(z_block, w)
    sign_w = np.sign(w)
    sign_w = sign_w.where(sign_w != 0.0, 1.0)
    sign_only = _score_matched(z_block, sign_w)
    ones = pd.Series(1.0, index=w.index)
    unweighted = _score_matched(z_block, ones)
    return {"exact_weight": exact, "sign_only": sign_only, "unweighted": unweighted}


def _p_from_null(observed: float, nulls: np.ndarray, alternative: str = "greater") -> dict[str, float]:
    finite = nulls[np.isfinite(nulls)]
    if not np.isfinite(observed) or len(finite) == 0:
        return {
            "null_n": int(len(finite)),
            "null_mean": np.nan,
            "null_sd": np.nan,
            "null_q95": np.nan,
            "p_null": np.nan,
            "percentile": np.nan,
        }
    if alternative == "greater":
        n_ge = int(np.sum(finite >= observed))
    else:
        n_ge = int(np.sum(np.abs(finite) >= abs(observed)))
    return {
        "null_n": int(len(finite)),
        "null_mean": float(np.mean(finite)),
        "null_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "null_q95": float(np.quantile(finite, 0.95)),
        "p_null": (n_ge + 1) / (len(finite) + 1),
        "percentile": float(100.0 * np.mean(finite < observed)),
    }


def _scifo_gene_table(
    out_dir: Path,
    organ: str,
    samples: pd.DataFrame,
    gene_of: pd.Series,
) -> pd.DataFrame:
    values, ages = _load_processed(out_dir, organ, samples)
    stats = _row_stats(values)
    stats["spearman_vs_age"] = _spearman_rows(values, ages)
    de_path = out_dir / "de_overlap" / f"{organ}_de.csv"
    de = pd.read_csv(de_path)
    de["accession"] = de["accession"].astype(str)
    stats["age_de"] = stats.index.isin(set(de.loc[de["age_de"].astype(bool), "accession"]))
    mem_path = out_dir / "modules" / f"{organ}_membership.csv"
    if mem_path.exists():
        mem = pd.read_csv(mem_path)
        mem["accession"] = mem["accession"].astype(str)
        stats["module"] = mem.drop_duplicates("accession").set_index("accession")["module"].reindex(stats.index)
        stats["module"] = stats["module"].fillna("unassigned")
    else:
        stats["module"] = "unassigned"
    return _collapse_gene(stats, gene_of)


def _rna_gene_stats(rna_log: pd.DataFrame) -> pd.DataFrame:
    stats = _row_stats(rna_log)
    stats["detected_frac"] = (rna_log.to_numpy(dtype=float) > np.log2(1.0 + 0.0)).mean(axis=1)
    return stats


def _prepare_organ(
    data_dir: Path,
    out_dir: Path,
    organ: str,
    metadata: pd.DataFrame,
    scifo_samples: pd.DataFrame,
    weights: pd.Series,
) -> dict:
    values, samples = load_whole_tissue(data_dir, organ, metadata)
    abundance = values.drop(columns=["description"], errors="ignore")
    acc_genes = load_accession_genes(data_dir, organ, metadata)
    gene_of = acc_genes.drop_duplicates("accession").set_index("accession")["gene"]
    rna = load_rna_matrix(data_dir, organ)
    samples = _align_samples(samples, rna)
    rna_genes = set(rna.index.astype(str))
    w_gene, protein_gene, gene_map = _gene_weights_and_protein(abundance, acc_genes, weights, rna_genes)
    matched = [g for g in w_gene.index if g in rna_genes]
    sample_ids = samples["sample_id"].tolist()
    rna_cols = samples["rna_col"].tolist()
    ages = samples["age_months"].to_numpy(dtype=float)
    prot = protein_gene.reindex(index=matched, columns=sample_ids)
    rna_matched = rna.reindex(index=matched, columns=rna_cols)
    rna_matched.columns = sample_ids
    rna_log_matched = np.log2(rna_matched.clip(lower=0.0) + 1.0)
    rna_all = np.log2(rna.clip(lower=0.0).reindex(columns=rna_cols) + 1.0)
    rna_all.columns = sample_ids
    w_use = w_gene.reindex(matched).astype(float)
    scifo = _scifo_gene_table(out_dir, organ, scifo_samples, gene_of)
    rna_stats = _rna_gene_stats(rna_all)
    tmt_stats = _row_stats(abundance)
    tmt_stats["spearman_vs_age"] = _spearman_rows(abundance, ages)
    tmt_gene = _collapse_gene(tmt_stats, gene_of)
    return {
        "organ": organ,
        "samples": samples,
        "ages": ages,
        "sample_ids": sample_ids,
        "w_use": w_use,
        "matched": matched,
        "prot": prot,
        "rna_log_matched": rna_log_matched,
        "rna_all": rna_all,
        "scifo": scifo,
        "rna_stats": rna_stats,
        "tmt_gene": tmt_gene,
        "gene_map": gene_map,
        "n_clock": int(len(weights)),
    }


def _pool_frame(bundle: dict) -> pd.DataFrame:
    clock = set(bundle["matched"])
    rna_stats = bundle["rna_stats"].copy()
    rna_stats = rna_stats.loc[~rna_stats.index.isin(clock)]
    scifo = bundle["scifo"]
    tmt = bundle["tmt_gene"]
    pool = rna_stats.rename(columns={"mean": "rna_mean", "variance": "rna_var", "observed_frac": "rna_obs"})
    if not scifo.empty:
        pool = pool.join(
            scifo.rename(
                columns={
                    "mean": "scifo_mean",
                    "variance": "scifo_var",
                    "observed_frac": "scifo_obs",
                    "spearman_vs_age": "scifo_rho",
                    "age_de": "age_de",
                    "module": "module",
                }
            ),
            how="left",
        )
    else:
        pool["scifo_mean"] = np.nan
        pool["scifo_rho"] = np.nan
        pool["age_de"] = False
        pool["module"] = "unassigned"
        pool["scifo_obs"] = np.nan
    if not tmt.empty:
        pool = pool.join(tmt[["mean", "spearman_vs_age"]].rename(columns={"mean": "tmt_mean", "spearman_vs_age": "tmt_rho"}), how="left")
    de = pool["age_de"]
    pool["age_de"] = de.eq(True) | de.eq(1)
    pool["module"] = pool["module"].fillna("unassigned")
    if "detected_frac" in pool.columns:
        pool["detected"] = pool["detected_frac"].fillna(0) > 0.5
    else:
        means = pool["rna_mean"].to_numpy(dtype=float)
        pool["detected"] = np.isfinite(means) & (means > 0)
    return pool


def _clock_covariates(bundle: dict, pool: pd.DataFrame) -> pd.DataFrame:
    genes = bundle["matched"]
    scifo = bundle["scifo"]
    rna_stats = bundle["rna_stats"]
    rows = []
    for gene in genes:
        row = {"gene": gene, "weight": float(bundle["w_use"].loc[gene])}
        if gene in rna_stats.index:
            row["rna_mean"] = float(rna_stats.loc[gene, "mean"])
            row["rna_var"] = float(rna_stats.loc[gene, "variance"])
            row["rna_obs"] = float(rna_stats.loc[gene, "observed_frac"])
        else:
            row["rna_mean"] = row["rna_var"] = row["rna_obs"] = np.nan
        if gene in scifo.index:
            row["scifo_rho"] = float(scifo.loc[gene, "spearman_vs_age"])
            row["scifo_obs"] = float(scifo.loc[gene, "observed_frac"])
            row["age_de"] = bool(scifo.loc[gene, "age_de"])
            row["module"] = str(scifo.loc[gene, "module"])
        else:
            row["scifo_rho"] = row["scifo_obs"] = np.nan
            row["age_de"] = False
            row["module"] = "unassigned"
        rows.append(row)
    return pd.DataFrame(rows).set_index("gene")


def _detected_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if "detected" in pool.columns:
        sub = pool.loc[pool["detected"].astype(bool)]
        if len(sub) >= 10:
            return sub
    return pool


def _proteome_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if "scifo_rho" not in pool.columns:
        return pool.iloc[0:0]
    return pool.loc[pool["scifo_rho"].notna()]


def _pool_for_null(kind: str, pool: pd.DataFrame) -> pd.DataFrame:
    if kind in {"random_detected", "abundance_matched", "variance_matched"}:
        return _detected_pool(pool)
    if kind in {"missingness_matched", "age_rho_matched", "de_status_matched", "module_matched"}:
        return _proteome_pool(pool)
    return pool


def _draw_null_genes(
    kind: str,
    clock_cov: pd.DataFrame,
    pool: pd.DataFrame,
    rng: np.random.Generator,
) -> list[str] | None:
    k = len(clock_cov)
    work = _pool_for_null(kind, pool)
    if kind == "random_detected":
        ids = work.index.to_numpy()
        if len(ids) < k:
            return None
        return [str(x) for x in rng.choice(ids, size=k, replace=False)]
    if kind == "abundance_matched":
        return _match_values(
            clock_cov["rna_mean"].to_numpy(dtype=float),
            work.index.to_numpy(),
            work["rna_mean"].to_numpy(dtype=float),
            rng,
        )
    if kind == "variance_matched":
        return _match_values(
            clock_cov["rna_var"].to_numpy(dtype=float),
            work.index.to_numpy(),
            work["rna_var"].to_numpy(dtype=float),
            rng,
        )
    if kind == "missingness_matched":
        col = "scifo_obs" if work["scifo_obs"].notna().sum() >= k else "rna_obs"
        clock_col = "scifo_obs" if col == "scifo_obs" else "rna_obs"
        ok = work[col].notna()
        return _match_values(
            clock_cov[clock_col].to_numpy(dtype=float),
            work.index[ok].to_numpy(),
            work.loc[ok, col].to_numpy(dtype=float),
            rng,
        )
    if kind == "age_rho_matched":
        ok = work["scifo_rho"].notna()
        if int(ok.sum()) < k:
            return None
        return _match_values(
            clock_cov["scifo_rho"].to_numpy(dtype=float),
            work.index[ok].to_numpy(),
            work.loc[ok, "scifo_rho"].to_numpy(dtype=float),
            rng,
        )
    if kind == "de_status_matched":
        picked = _match_discrete(
            clock_cov["age_de"].astype(bool).to_numpy(),
            work.index.to_numpy(),
            work["age_de"].astype(bool).to_numpy(),
            rng,
        )
        if picked is not None:
            return picked
        n_de = int(clock_cov["age_de"].astype(bool).sum())
        n_not = k - n_de
        de_ids = work.index[work["age_de"].astype(bool)].to_numpy()
        ot_ids = work.index[~work["age_de"].astype(bool)].to_numpy()
        if len(de_ids) == 0 and len(ot_ids) == 0:
            return None
        de_take = min(n_de, len(de_ids))
        ot_take = min(n_not, len(ot_ids))
        chosen = []
        if de_take:
            chosen.extend(rng.choice(de_ids, size=de_take, replace=False).tolist())
        if ot_take:
            chosen.extend(rng.choice(ot_ids, size=ot_take, replace=False).tolist())
        leftover = k - len(chosen)
        if leftover > 0:
            rest = work.index.difference(pd.Index(chosen)).to_numpy()
            if len(rest) < leftover:
                return None
            chosen.extend(rng.choice(rest, size=leftover, replace=False).tolist())
        return [str(x) for x in chosen[:k]]
    if kind == "module_matched":
        picked = _match_discrete(
            clock_cov["module"].astype(str).to_numpy(),
            work.index.to_numpy(),
            work["module"].astype(str).to_numpy(),
            rng,
        )
        if picked is not None:
            return picked
        mods = set(clock_cov["module"].astype(str))
        sub = work.loc[work["module"].astype(str).isin(mods)]
        ids = (sub if len(sub) >= k else work).index.to_numpy()
        if len(ids) < k:
            return None
        return [str(x) for x in rng.choice(ids, size=k, replace=False)]
    return None


def _gene_index_map(index) -> dict[str, int]:
    return {str(g): i for i, g in enumerate(index)}


def _null_rhos_gene_sets(
    kind: str,
    bundle: dict,
    clock_cov: pd.DataFrame,
    pool: pd.DataFrame,
    n_null: int,
) -> np.ndarray:
    ages = bundle["ages"]
    w_vals = clock_cov["weight"].to_numpy(dtype=float)
    genes_clock = clock_cov.index.tolist()
    rna_all = bundle["rna_all"]
    rna_mat = rna_all.to_numpy(dtype=float)
    idx_of = _gene_index_map(rna_all.index)
    matched_mat = bundle["rna_log_matched"].reindex(genes_clock).to_numpy(dtype=float)
    rhos = np.full(n_null, np.nan)
    for i in range(n_null):
        rng = np.random.default_rng(int(RNG.integers(0, 2**31 - 1)))
        if kind == "weight_permuted":
            rhos[i] = _score_rho_np(matched_mat, rng.permutation(w_vals), ages)
            continue
        if kind == "sign_permuted":
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(w_vals))
            rhos[i] = _score_rho_np(matched_mat, np.abs(w_vals) * signs, ages)
            continue
        picked = _draw_null_genes(kind, clock_cov, pool, rng)
        if picked is None:
            continue
        rows = []
        w_use = []
        for gene, weight in zip(picked, w_vals):
            if gene in idx_of:
                rows.append(idx_of[gene])
                w_use.append(weight)
        if len(rows) < MIN_MATCHED:
            continue
        rhos[i] = _score_rho_np(rna_mat[np.asarray(rows, dtype=int)], np.asarray(w_use, dtype=float), ages)
    return rhos


def analyze_organ_portability(bundle: dict) -> dict[str, pd.DataFrame]:
    organ = bundle["organ"]
    ages = bundle["ages"]
    w = bundle["w_use"]
    rna_m = bundle["rna_log_matched"]
    prot = bundle["prot"]
    pool = _pool_frame(bundle)
    clock_cov = _clock_covariates(bundle, pool)
    variants_rna = _score_variants(rna_m, w)
    variants_prot = _score_variants(prot, w)
    score_rows = []
    for layer, variants in (("rna", variants_rna), ("protein", variants_prot)):
        for name, series in variants.items():
            vals = series.reindex(bundle["sample_ids"]).to_numpy(dtype=float)
            r, p, pp = permutation_spearman(vals, ages)
            score_rows.append(
                {
                    "organ": organ,
                    "layer": layer,
                    "score": name,
                    "spearman_vs_age": r,
                    "p_parametric": p,
                    "p_permutation": pp,
                    "pearson_vs_age": _pearson(vals, ages),
                    "n_genes": int(len(w)),
                    "n_samples": int(len(ages)),
                }
            )
    scores_long = pd.DataFrame(score_rows)
    observed_rna = float(scores_long.loc[(scores_long["layer"] == "rna") & (scores_long["score"] == "exact_weight"), "spearman_vs_age"].iloc[0])

    null_summary = []
    null_draws = []
    for kind in NULL_ORDER:
        rhos = _null_rhos_gene_sets(kind, bundle, clock_cov, pool, N_NULL)
        stats = _p_from_null(observed_rna, rhos)
        n_pool = int(len(_pool_for_null(kind, pool))) if kind not in {"weight_permuted", "sign_permuted"} else int(len(clock_cov))
        null_summary.append(
            {
                "organ": organ,
                "null": kind,
                "observed_spearman_rna_vs_age": observed_rna,
                **stats,
                "n_clock_genes": int(len(w)),
                "n_pool": n_pool,
            }
        )
        null_draws.append(pd.DataFrame({"organ": organ, "null": kind, "draw": np.arange(len(rhos)), "spearman_rna_vs_age": rhos}))
        hit = null_summary[-1]
        ptxt = f"{hit['p_null']:.4g}" if np.isfinite(hit["p_null"]) else "NA"
        mean = hit["null_mean"] if np.isfinite(hit["null_mean"]) else float("nan")
        sd = hit.get("null_sd", float("nan"))
        sd = sd if np.isfinite(sd) else float("nan")
        print(f"    {kind:22s}  p={ptxt}  null mean={mean:.3f}  sd={sd:.3f}  n_pool={n_pool}", flush=True)

    concordance = _concordance_tables(bundle, pool, clock_cov)
    return {
        "score_anatomy": scores_long,
        "null_summary": pd.DataFrame(null_summary),
        "null_draws": pd.concat(null_draws, ignore_index=True),
        "clock_covariates": clock_cov.reset_index().assign(organ=organ),
        **concordance,
    }


def _sign_agree(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if mask.sum() < 3:
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


def _slope_concordance(rho_p: np.ndarray, rho_r: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(rho_p) & np.isfinite(rho_r)
    n = int(mask.sum())
    if n < 5:
        return {
            "n": n,
            "spearman_slopes": np.nan,
            "pearson_slopes": np.nan,
            "sign_agreement": np.nan,
            "frac_concordant_strong": np.nan,
        }
    r_s, _ = spearmanr(rho_p[mask], rho_r[mask])
    r_p = _pearson(rho_p[mask], rho_r[mask])
    strong = (np.abs(rho_p[mask]) >= RHO_CUT) | (np.abs(rho_r[mask]) >= RHO_CUT)
    if strong.sum() >= 3:
        frac = float(np.mean(np.sign(rho_p[mask][strong]) == np.sign(rho_r[mask][strong])))
    else:
        frac = np.nan
    return {
        "n": n,
        "spearman_slopes": float(r_s) if np.isfinite(r_s) else np.nan,
        "pearson_slopes": float(r_p) if np.isfinite(r_p) else np.nan,
        "sign_agreement": _sign_agree(rho_p, rho_r),
        "frac_concordant_strong": frac,
    }


def _all_gene_slopes(bundle: dict) -> pd.DataFrame:
    """Takasugi protein vs RNA age slopes for every gene in both layers."""
    ages = bundle["ages"]
    sample_ids = bundle["sample_ids"]
    rna_all = bundle["rna_all"]
    scifo = bundle["scifo"]
    clock = set(bundle["matched"])
    abundance_by_gene = bundle.get("prot_all", bundle["prot"])
    genes = [g for g in abundance_by_gene.index if g in rna_all.index]
    rows = []
    for gene in genes:
        p_vec = abundance_by_gene.loc[gene, sample_ids].to_numpy(dtype=float)
        r_vec = rna_all.loc[gene, sample_ids].to_numpy(dtype=float)
        rp, _ = _spearman_pair(p_vec, ages)
        rr, _ = _spearman_pair(r_vec, ages)
        if gene in scifo.index:
            rs = float(scifo.loc[gene, "spearman_vs_age"])
            de = bool(scifo.loc[gene, "age_de"])
        else:
            rs = np.nan
            de = False
        if gene in clock:
            group = "clock"
        elif de:
            group = "age_de"
        else:
            group = "other"
        rows.append(
            {
                "organ": bundle["organ"],
                "gene": gene,
                "group": group,
                "age_de": de,
                "clock": gene in clock,
                "spearman_tmt_vs_age": rp,
                "spearman_rna_vs_age": rr,
                "spearman_scifo_vs_age": rs,
                "same_sign_tmt_rna": int(
                    np.isfinite(rp) and np.isfinite(rr) and rp != 0 and rr != 0 and np.sign(rp) == np.sign(rr)
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_full_tmt_gene_matrix(
    bundle: dict,
    data_dir: Path,
    organ: str,
    metadata: pd.DataFrame,
) -> dict:
    values, samples = load_whole_tissue(data_dir, organ, metadata)
    abundance = values.drop(columns=["description"], errors="ignore")
    acc_genes = load_accession_genes(data_dir, organ, metadata)
    gene_of = acc_genes.drop_duplicates("accession").set_index("accession")["gene"]
    rna_genes = set(bundle["rna_all"].index.astype(str))
    blocks: dict[str, list[str]] = {}
    for acc in abundance.index.astype(str):
        gene = str(gene_of[acc]) if acc in gene_of.index else ""
        if not gene or gene not in rna_genes:
            continue
        blocks.setdefault(gene, []).append(acc)
    if not blocks:
        bundle["prot_all"] = bundle["prot"]
        return bundle
    prot_all = pd.DataFrame({gene: abundance.loc[accs, bundle["sample_ids"]].median(axis=0) for gene, accs in blocks.items()}).T
    bundle["prot_all"] = prot_all
    return bundle


def _concordance_tables(bundle: dict, pool: pd.DataFrame, clock_cov: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slopes = _all_gene_slopes(bundle)
    organ = bundle["organ"]
    summary_rows = []
    for group, sub in (
        ("all", slopes),
        ("age_de", slopes.loc[slopes["age_de"]]),
        ("clock", slopes.loc[slopes["clock"]]),
        ("other", slopes.loc[slopes["group"] == "other"]),
    ):
        stats = _slope_concordance(
            sub["spearman_tmt_vs_age"].to_numpy(dtype=float),
            sub["spearman_rna_vs_age"].to_numpy(dtype=float),
        )
        summary_rows.append({"organ": organ, "set": group, **stats})
        if group == "clock":
            stats_scifo = _slope_concordance(
                sub["spearman_scifo_vs_age"].to_numpy(dtype=float),
                sub["spearman_rna_vs_age"].to_numpy(dtype=float),
            )
            summary_rows.append({"organ": organ, "set": "clock_scifo_vs_takasugi_rna", **stats_scifo})

    clock_obs = next(r for r in summary_rows if r["set"] == "clock")
    matched_null = np.full(N_NULL, np.nan)
    clock_genes = bundle["matched"]
    k = len(clock_genes)
    rng = np.random.default_rng(2)
    for i in range(N_NULL):
        picked = _draw_null_genes("age_rho_matched", clock_cov, pool, rng)
        if picked is None:
            picked = _draw_null_genes("abundance_matched", clock_cov, pool, rng)
        if picked is None:
            continue
        sub = slopes.loc[slopes["gene"].isin(picked)]
        stats = _slope_concordance(
            sub["spearman_tmt_vs_age"].to_numpy(dtype=float),
            sub["spearman_rna_vs_age"].to_numpy(dtype=float),
        )
        matched_null[i] = stats["spearman_slopes"]
    null_stats = _p_from_null(float(clock_obs["spearman_slopes"]), matched_null)
    summary_rows.append(
        {
            "organ": organ,
            "set": "clock_vs_age_rho_matched_null",
            "n": k,
            "spearman_slopes": clock_obs["spearman_slopes"],
            "pearson_slopes": clock_obs["pearson_slopes"],
            "sign_agreement": clock_obs["sign_agreement"],
            "frac_concordant_strong": clock_obs["frac_concordant_strong"],
            **{f"null_{k}": v for k, v in null_stats.items()},
        }
    )
    return {
        "gene_slopes": slopes,
        "concordance": pd.DataFrame(summary_rows),
        "concordance_null_draws": pd.DataFrame(
            {"organ": organ, "draw": np.arange(len(matched_null)), "spearman_slopes": matched_null}
        ),
    }


def decompose_lu(out_dir: Path, data_dir: Path) -> pd.DataFrame:
    matrix_path = data_dir / "Aging_mouse_organ_PulseDIA_Matrix.tsv.xz"
    if not matrix_path.exists():
        return pd.DataFrame()
    print("  Lu aging males: gene / sign / weight anatomy ...", flush=True)
    values = load_lu_matrix(data_dir)
    meta = load_lu_metadata(data_dir)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    rows = []
    aging = meta.loc[(meta["Type"].astype(str) == "Aging") & (meta["Sex"].astype(str) == "Male")].copy()
    for organ, lu_name in LU_ORGAN_MAP.items():
        subset = aging.loc[aging["lu_organ"] == lu_name]
        if subset.empty:
            continue
        w = _frozen_weights(stab_dir, coef_dir, organ)["stable"]
        cols = [c for c in subset["sample_id"].tolist() if c in values.columns]
        overlap = [acc for acc in w.index.astype(str) if acc in values.index]
        if len(overlap) < MIN_MATCHED or len(cols) < 5:
            continue
        block = values.loc[overlap, cols]
        ages = subset.set_index("sample_id").loc[cols, "age_months"].to_numpy(dtype=float)
        w_use = w.reindex(overlap).astype(float)
        block_np = block.reindex(index=w_use.index, columns=cols).to_numpy(dtype=float)
        variants = _score_variants(block, w_use)
        rng = np.random.default_rng(3)
        w_vals = w_use.to_numpy(dtype=float)
        perm_w = np.array([_score_rho_np(block_np, rng.permutation(w_vals), ages) for _ in range(N_NULL)])
        perm_s = np.array(
            [
                _score_rho_np(block_np, np.abs(w_vals) * rng.choice(np.array([-1.0, 1.0]), size=len(w_vals)), ages)
                for _ in range(N_NULL)
            ]
        )
        got = {}
        for name, series in variants.items():
            vals = series.reindex(cols).to_numpy(dtype=float)
            r, p = _spearman_pair(vals, ages)
            extra = {}
            if name == "exact_weight":
                extra.update({f"weight_perm_{k}": v for k, v in _p_from_null(r, perm_w).items()})
                extra.update({f"sign_perm_{k}": v for k, v in _p_from_null(r, perm_s).items()})
            row = {
                "organ": organ,
                "lu_organ": lu_name,
                "layer": "lu_pulseDIA",
                "score": name,
                "spearman_vs_age": r,
                "p_parametric": p,
                "n_genes": int(len(overlap)),
                "n_samples": int(len(cols)),
                **extra,
            }
            rows.append(row)
            got[name] = r
        print(
            f"    {organ}: exact ρ={got.get('exact_weight', float('nan')):.3f}  "
            f"sign ρ={got.get('sign_only', float('nan')):.3f}  "
            f"unweighted ρ={got.get('unweighted', float('nan')):.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def write_rna_portability(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "rna_portability"
    dest.mkdir(parents=True, exist_ok=True)
    metadata = load_takasugi_metadata(data_dir)
    scifo_samples = pd.read_csv(out_dir / "metadata" / "samples.csv")
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    buckets: dict[str, list[pd.DataFrame]] = {
        "score_anatomy": [],
        "null_summary": [],
        "null_draws": [],
        "clock_covariates": [],
        "gene_slopes": [],
        "concordance": [],
        "concordance_null_draws": [],
    }
    for organ in TRANSFER_ORGANS:
        print(f"  Portability {organ} ...", flush=True)
        weights = _frozen_weights(stab_dir, coef_dir, organ)["stable"]
        bundle = _prepare_organ(data_dir, out_dir, organ, metadata, scifo_samples, weights)
        if len(bundle["matched"]) < MIN_MATCHED:
            print(f"    skip: {len(bundle['matched'])} matched genes", flush=True)
            continue
        bundle = attach_full_tmt_gene_matrix(bundle, data_dir, organ, metadata)
        tables = analyze_organ_portability(bundle)
        for key, frame in tables.items():
            buckets[key].append(frame)
    out = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for name, frames in buckets.items()}
    lu_dir = data_dir.parent / "Lu_intervention"
    if not lu_dir.exists():
        lu_dir = project_root() / "Dataset" / "Lu_intervention"
    out["lu_anatomy"] = decompose_lu(out_dir, lu_dir)
    out["score_anatomy"].to_csv(dest / "score_anatomy.csv", index=False)
    out["null_summary"].to_csv(dest / "null_summary.csv", index=False)
    out["null_draws"].to_csv(dest / "null_draws.csv", index=False)
    out["clock_covariates"].to_csv(dest / "clock_covariates.csv", index=False)
    out["gene_slopes"].to_csv(dest / "gene_slopes.csv", index=False)
    out["concordance"].to_csv(dest / "concordance.csv", index=False)
    out["concordance_null_draws"].to_csv(dest / "concordance_null_draws.csv", index=False)
    if not out["lu_anatomy"].empty:
        out["lu_anatomy"].to_csv(dest / "lu_score_anatomy.csv", index=False)
    return out
