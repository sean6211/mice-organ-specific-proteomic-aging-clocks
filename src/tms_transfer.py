"""Frozen Scifo clocks on Tabula Muris Senis bulk RNA (GSE132040). No refit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ageaccel import permutation_spearman
from .rna_cross_organ import _clock_gene_weights
from .rna_portability import N_NULL, _p_from_null, _score_rho_np, _score_variants
from .shared_program import _pearson
from .takasugi_rna import MIN_MATCHED, _normalize_gene, load_accession_genes
from .takasugi_transfer import WHOLE_TISSUE, _frozen_weights, load_takasugi_metadata

TMS_ORGAN = {
    "Brain": "brain",
    "Heart": "heart",
    "Lung": "lung",
    "Liver": "liver",
    "Kidney": "kidney",
    "Limb_Muscle": "muscle",
    "Spleen": "spleen",
}
PANEL_ORGANS = ["brain", "heart", "lung", "liver", "kidney", "spleen", "muscle"]
RNG = np.random.default_rng(11)


def _mouse_symbol(gene: object) -> str:
    token = _normalize_gene(gene)
    if not token:
        return ""
    if token != token.upper():
        return token
    return token[0] + token[1:].lower()


def _default_tms_dir(root: Path) -> Path:
    named = root / "Dataset" / "GSE132040_Tabula_Menis"
    if named.exists():
        return named
    return root / "Dataset" / "GSE132040"


def load_tms_metadata(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "GSE132040_MACA_Bulk_metadata.csv.gz"
    meta = pd.read_csv(path)
    meta["sample_id"] = meta["Sample name"].astype(str)
    meta["tms_organ"] = meta["source name"].astype(str).str.replace(r"_\d+$", "", regex=True)
    meta["organ"] = meta["tms_organ"].map(TMS_ORGAN)
    meta["age_months"] = pd.to_numeric(meta["characteristics: age"], errors="coerce")
    sex = meta["characteristics: sex"].astype(str).str.lower()
    meta["sex"] = sex.map({"m": "male", "f": "female"}).fillna("missing")
    return meta


def load_tms_matrix(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("GSE132040_*AHH3J3DSXX*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No TMS count matrix under {data_dir}")
    rna = pd.read_csv(files[0], index_col=0)
    rna.index = [_normalize_gene(g) for g in rna.index]
    rna = rna.groupby(rna.index).median()
    rna.columns = [str(c).replace(".gencode.vM19", "") for c in rna.columns]
    return rna.clip(lower=0.0)


def _lookup_index(rna_index) -> dict[str, str]:
    out = {}
    for gene in rna_index.astype(str):
        out[gene] = gene
        out[gene.lower()] = gene
    return out


def _gene_weights_for_organ(
    organ: str,
    weights: pd.Series,
    takasugi_dir: Path,
    takasugi_meta: pd.DataFrame,
    stable_genes: pd.DataFrame,
) -> pd.Series:
    pooled: dict[str, float] = {}
    if organ in WHOLE_TISSUE:
        acc_genes = load_accession_genes(takasugi_dir, organ, takasugi_meta)
        mapped = _clock_gene_weights(acc_genes, weights)
        for gene, w in mapped.items():
            pooled[str(gene)] = pooled.get(str(gene), 0.0) + float(w)
        mapped_acc = set(acc_genes["accession"].astype(str))
    else:
        mapped_acc = set()
    sub = stable_genes.loc[stable_genes["organ"] == organ]
    gene_of = sub.drop_duplicates("accession").set_index("accession")["gene"]
    for acc, w in weights.items():
        acc = str(acc)
        if acc in mapped_acc and pooled:
            continue
        gene = _mouse_symbol(gene_of[acc]) if acc in gene_of.index else ""
        if not gene:
            continue
        pooled[gene] = pooled.get(gene, 0.0) + float(w)
    return pd.Series(pooled, dtype=float)


def _subset_samples(meta: pd.DataFrame, organ: str, sex: str) -> pd.DataFrame:
    sub = meta.loc[meta["organ"] == organ].copy()
    sub = sub.loc[np.isfinite(sub["age_months"])]
    if sex != "both":
        sub = sub.loc[sub["sex"] == sex]
    return sub.reset_index(drop=True)


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    return _pearson(x[mask], y[mask])


def _metric_row(
    organ: str,
    sex: str,
    score: str,
    vals: np.ndarray,
    ages: np.ndarray,
    n_clock: int,
    n_matched: int,
    note: str = "",
    permute: bool = False,
) -> dict:
    if permute:
        r, p, pp = permutation_spearman(vals, ages)
    else:
        r, p = _spearman_pair_safe(vals, ages)
        pp = np.nan
    return {
        "organ": organ,
        "sex": sex,
        "score": score,
        "spearman_vs_age": r,
        "p_parametric": p,
        "p_permutation": pp,
        "pearson_vs_age": _pearson_safe(vals, ages),
        "n_clock_genes": n_clock,
        "n_matched": n_matched,
        "n_samples": int(np.isfinite(ages).sum()),
        "age_min": float(np.nanmin(ages)),
        "age_max": float(np.nanmax(ages)),
        "note": note,
    }


def write_tms_transfer(root: Path, out_dir: Path, tms_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    dest = out_dir / "tms_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    tms_dir = tms_dir or _default_tms_dir(root)
    takasugi_dir = root / "Dataset" / "Takasugi"
    print(f"  Loading TMS bulk from {tms_dir} ...", flush=True)
    meta = load_tms_metadata(tms_dir)
    rna = load_tms_matrix(tms_dir)
    lookup = _lookup_index(rna.index)
    takasugi_meta = load_takasugi_metadata(takasugi_dir)
    stable_genes = pd.read_csv(out_dir / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    organs = [o for o in PANEL_ORGANS if (stab_dir / f"{o}.csv").exists()]
    score_rows = []
    metric_rows = []
    null_rows = []
    for organ in organs:
        print(f"  TMS {organ} ...", flush=True)
        weights = _frozen_weights(stab_dir, coef_dir, organ)["stable"]
        w_gene = _gene_weights_for_organ(organ, weights, takasugi_dir, takasugi_meta, stable_genes)
        mapped_w: dict[str, float] = {}
        for gene, w in w_gene.items():
            hit = lookup.get(str(gene)) or lookup.get(str(gene).lower())
            if hit:
                mapped_w[hit] = mapped_w.get(hit, 0.0) + float(w)
        w_use = pd.Series(mapped_w, dtype=float)
        n_clock = int(len(w_gene))
        n_matched = int(len(w_use))
        for sex in ("male", "female", "both"):
            samples = _subset_samples(meta, organ, sex)
            cols = [c for c in samples["sample_id"] if c in rna.columns]
            samples = samples.loc[samples["sample_id"].isin(cols)].copy()
            if len(samples) < 8 or n_matched < MIN_MATCHED:
                metric_rows.append(
                    {
                        "organ": organ,
                        "sex": sex,
                        "n_clock_genes": n_clock,
                        "n_matched": n_matched,
                        "n_samples": int(len(samples)),
                        "note": "too_few_samples_or_genes",
                    }
                )
                continue
            ages = samples["age_months"].to_numpy(dtype=float)
            ids = samples["sample_id"].tolist()
            block = np.log2(rna.reindex(index=w_use.index, columns=ids).clip(lower=0.0) + 1.0)
            variants = _score_variants(block, w_use)
            got = {}
            for name, series in variants.items():
                vals = series.reindex(ids).to_numpy(dtype=float)
                permute = sex == "male" and name == "exact_weight"
                row = _metric_row(organ, sex, name, vals, ages, n_clock, n_matched, permute=permute)
                got[name] = row["spearman_vs_age"]
                metric_rows.append(row)
                if name == "exact_weight":
                    tmp = samples[["sample_id", "organ", "age_months", "sex", "tms_organ"]].copy()
                    tmp["score"] = vals
                    tmp["n_matched"] = n_matched
                    score_rows.append(tmp)
            if sex == "male":
                samples3 = samples.loc[samples["age_months"] >= 3]
                if len(samples3) >= 8:
                    ids3 = samples3["sample_id"].tolist()
                    ages3 = samples3["age_months"].to_numpy(dtype=float)
                    block3 = np.log2(rna.reindex(index=w_use.index, columns=ids3).clip(lower=0.0) + 1.0)
                    for name, series in _score_variants(block3, w_use).items():
                        vals3 = series.reindex(ids3).to_numpy(dtype=float)
                        metric_rows.append(
                            _metric_row(organ, sex, name, vals3, ages3, n_clock, n_matched, note="age_ge_3")
                        )
                print(
                    f"    male n={len(ids)}  genes={n_matched}/{n_clock}  "
                    f"exact ρ={got.get('exact_weight', float('nan')):.3f}  "
                    f"sign ρ={got.get('sign_only', float('nan')):.3f}  "
                    f"unweighted ρ={got.get('unweighted', float('nan')):.3f}",
                    flush=True,
                )
                rna_log_all = np.log2(rna.reindex(columns=ids).clip(lower=0.0) + 1.0)
                detected = rna_log_all.index[(rna_log_all.to_numpy(dtype=float) > 0).mean(axis=1) > 0.5]
                detected = np.array([g for g in detected if g not in set(w_use.index)])
                k = int(len(w_use))
                w_vals = w_use.to_numpy(dtype=float)
                sign_vals = np.sign(w_vals)
                sign_vals = np.where(sign_vals == 0.0, 1.0, sign_vals)
                idx_of = {str(g): i for i, g in enumerate(rna_log_all.index.astype(str))}
                mat = rna_log_all.to_numpy(dtype=float)
                exact_null = np.full(N_NULL, np.nan)
                sign_null = np.full(N_NULL, np.nan)
                for i in range(N_NULL):
                    if len(detected) < k:
                        break
                    pick = RNG.choice(detected, size=k, replace=False)
                    rows = [idx_of[str(g)] for g in pick if str(g) in idx_of]
                    if len(rows) < MIN_MATCHED:
                        continue
                    row_idx = np.asarray(rows, dtype=int)
                    w_use_i = w_vals[: len(row_idx)]
                    exact_null[i] = _score_rho_np(mat[row_idx], w_use_i, ages)
                    sign_null[i] = _score_rho_np(mat[row_idx], sign_vals[: len(row_idx)], ages)
                for score_name, obs, draws in (
                    ("exact_weight", float(got.get("exact_weight", np.nan)), exact_null),
                    ("sign_only", float(got.get("sign_only", np.nan)), sign_null),
                ):
                    stats = _p_from_null(obs, draws)
                    null_rows.append(
                        {
                            "organ": organ,
                            "sex": "male",
                            "score": score_name,
                            "null": "random_detected",
                            "observed": obs,
                            **stats,
                            "n_matched": n_matched,
                        }
                    )
                    ptxt = f"{stats['p_null']:.4g}" if np.isfinite(stats["p_null"]) else "NA"
                    print(
                        f"    {score_name} random-gene null  p={ptxt}  null mean={stats['null_mean']:.3f}",
                        flush=True,
                    )
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    nulls = pd.DataFrame(null_rows)
    summary = _male_summary(metrics, nulls)
    scores.to_csv(dest / "scores.csv", index=False)
    metrics.to_csv(dest / "metrics.csv", index=False)
    nulls.to_csv(dest / "null_summary.csv", index=False)
    summary.to_csv(dest / "summary.csv", index=False)
    return {"scores": scores, "metrics": metrics, "nulls": nulls, "summary": summary}


def _male_summary(metrics: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    primary = metrics.loc[(metrics["sex"] == "male") & (metrics["note"].fillna("") == "")]
    rows = []
    for organ, sub in primary.groupby("organ"):
        item = {"organ": organ}
        for _, row in sub.iterrows():
            item[row["score"]] = row["spearman_vs_age"]
            if row["score"] == "exact_weight":
                item["n_matched"] = row["n_matched"]
                item["n_clock_genes"] = row["n_clock_genes"]
                item["n_samples"] = row["n_samples"]
                item["p_permutation"] = row["p_permutation"]
        if not nulls.empty:
            score_col = nulls["score"] if "score" in nulls.columns else None
            for score_name, mean_key, p_key in (
                ("exact_weight", "null_mean", "p_null"),
                ("sign_only", "sign_null_mean", "sign_p_null"),
            ):
                if score_col is None:
                    hit = nulls.loc[nulls["organ"] == organ] if score_name == "exact_weight" else pd.DataFrame()
                else:
                    hit = nulls.loc[(nulls["organ"] == organ) & (nulls["score"] == score_name)]
                if not hit.empty:
                    n0 = hit.iloc[0]
                    item[mean_key] = n0.get("null_mean", np.nan)
                    item[p_key] = n0.get("p_null", np.nan)
        rows.append(item)
    order = {o: i for i, o in enumerate(PANEL_ORGANS)}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_ord"] = out["organ"].map(order)
        out = out.sort_values("_ord").drop(columns="_ord")
    return out.reset_index(drop=True)


def _spearman_pair_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from .takasugi_rna import _spearman_pair

    return _spearman_pair(x, y)
