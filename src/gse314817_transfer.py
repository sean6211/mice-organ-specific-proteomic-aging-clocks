"""Frozen Scifo clocks on Shavlakadze/Regeneron mouse bulk RNA (GSE314817). No refit.

C57BL/6J, both sexes, ages 6–27 months. Matching tissues only:
kidney, liver, spleen, left ventricle → heart, cortex → brain.
Lung is absent. Muscle lives in a separate archive without sample metadata here.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .rna_portability import N_NULL, _p_from_null, _score_rho_np, _score_variants
from .takasugi_rna import MIN_MATCHED, _normalize_gene
from .takasugi_transfer import _frozen_weights, load_takasugi_metadata
from .tms_transfer import (
    _gene_weights_for_organ,
    _lookup_index,
    _male_summary,
    _metric_row,
)

TISSUE_TO_ORGAN = {
    "kidney": "kidney",
    "liver": "liver",
    "spleen": "spleen",
    "left ventricle": "heart",
    "cortex": "brain",
}
PANEL_ORGANS = ["brain", "heart", "liver", "kidney", "spleen"]
RNG = np.random.default_rng(17)


def _default_data_dir(root: Path) -> Path:
    return root / "Dataset" / "GSE314817"


def load_gse314817_metadata(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "GSE314817_series_matrix.txt.gz"
    titles = []
    geos = []
    char_rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                geos = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                char_rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
    if not titles or len(char_rows) < 4:
        raise ValueError(f"Could not parse series matrix {path}")
    fields = {}
    for row in char_rows:
        key = row[0].split(":", 1)[0].strip().lower()
        fields[key] = [cell.split(":", 1)[-1].strip() for cell in row]
    sex_raw = pd.Series(fields["sex"]).str.lower()
    age_raw = pd.Series(fields["age"]).str.extract(r"(\d+)", expand=False)
    meta = pd.DataFrame(
        {
            "sample_id": [t.rsplit("_", 1)[-1] for t in titles],
            "title": titles,
            "geo": geos if geos else [""] * len(titles),
            "tissue": fields["tissue"],
            "strain": fields.get("strain", ["C57BL/6J"] * len(titles)),
            "sex": sex_raw.map({"male": "male", "female": "female"}).fillna("missing"),
            "age_months": pd.to_numeric(age_raw, errors="coerce"),
        }
    )
    meta["organ"] = meta["tissue"].map(TISSUE_TO_ORGAN)
    return meta


def load_gse314817_matrix(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "GSE314817_mouse_raw_counts_wo_muscle.txt.gz"
    rna = pd.read_csv(path, sep="\t", index_col=0)
    rna.index = [_normalize_gene(g) for g in rna.index]
    rna = rna.groupby(rna.index).median()
    rna.columns = [str(c) for c in rna.columns]
    return rna.clip(lower=0.0)


def _subset_samples(meta: pd.DataFrame, organ: str, sex: str) -> pd.DataFrame:
    sub = meta.loc[meta["organ"] == organ].copy()
    sub = sub.loc[np.isfinite(sub["age_months"])]
    if sex != "both":
        sub = sub.loc[sub["sex"] == sex]
    return sub.reset_index(drop=True)


def write_gse314817_transfer(root: Path, out_dir: Path, data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    dest = out_dir / "gse314817_transfer"
    dest.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or _default_data_dir(root)
    takasugi_dir = root / "Dataset" / "Takasugi"
    print(f"  Loading GSE314817 from {data_dir} ...", flush=True)
    meta = load_gse314817_metadata(data_dir)
    rna = load_gse314817_matrix(data_dir)
    lookup = _lookup_index(rna.index)
    takasugi_meta = load_takasugi_metadata(takasugi_dir)
    stable_genes = pd.read_csv(out_dir / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    organs = [o for o in PANEL_ORGANS if (stab_dir / f"{o}.csv").exists() and (meta["organ"] == o).any()]
    score_rows = []
    metric_rows = []
    null_rows = []
    for organ in organs:
        print(f"  GSE314817 {organ} ...", flush=True)
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
        tissue = meta.loc[meta["organ"] == organ, "tissue"].iloc[0]
        for sex in ("male", "female", "both"):
            samples = _subset_samples(meta, organ, sex)
            cols = [c for c in samples["sample_id"] if c in rna.columns]
            samples = samples.loc[samples["sample_id"].isin(cols)].copy()
            if len(samples) < 8 or n_matched < MIN_MATCHED:
                metric_rows.append(
                    {
                        "organ": organ,
                        "tissue": tissue,
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
                row["tissue"] = tissue
                got[name] = row["spearman_vs_age"]
                metric_rows.append(row)
                if name == "exact_weight":
                    tmp = samples[["sample_id", "organ", "tissue", "age_months", "sex"]].copy()
                    tmp["score"] = vals
                    tmp["n_matched"] = n_matched
                    score_rows.append(tmp)
            if sex != "male":
                continue
            print(
                f"    male n={len(ids)}  {tissue}  genes={n_matched}/{n_clock}  "
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
                exact_null[i] = _score_rho_np(mat[row_idx], w_vals[: len(row_idx)], ages)
                sign_null[i] = _score_rho_np(mat[row_idx], sign_vals[: len(row_idx)], ages)
            for score_name, obs, draws in (
                ("exact_weight", float(got.get("exact_weight", np.nan)), exact_null),
                ("sign_only", float(got.get("sign_only", np.nan)), sign_null),
            ):
                stats = _p_from_null(obs, draws)
                null_rows.append(
                    {
                        "organ": organ,
                        "tissue": tissue,
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
    if not summary.empty and "tissue" not in summary.columns:
        tissue_of = meta.dropna(subset=["organ"]).drop_duplicates("organ").set_index("organ")["tissue"]
        summary["tissue"] = summary["organ"].map(tissue_of)
    scores.to_csv(dest / "scores.csv", index=False)
    metrics.to_csv(dest / "metrics.csv", index=False)
    nulls.to_csv(dest / "null_summary.csv", index=False)
    summary.to_csv(dest / "summary.csv", index=False)
    return {"scores": scores, "metrics": metrics, "nulls": nulls, "summary": summary}
