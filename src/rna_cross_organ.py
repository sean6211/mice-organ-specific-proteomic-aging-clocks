"""Cross-organ RNA projection of frozen Scifo clocks. No refit.

Clock organ A gene-level weights scored on Takasugi RNA from organ B.
Same-organ cells should recover the published RNA rank; cross-organ cells
test whether that rank is an organ-private barcode or a shared aging
transcriptome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ageaccel import permutation_spearman
from .shared_program import _pearson
from .takasugi_rna import (
    MIN_MATCHED,
    RNA_FILES,
    _align_samples,
    _normalize_gene,
    _score_matched,
    load_accession_genes,
    load_rna_matrix,
)
from .takasugi_transfer import (
    TRANSFER_ORGANS,
    _frozen_weights,
    load_takasugi_metadata,
    load_whole_tissue,
)

PANEL_ORGANS = [o for o in ["brain", "heart", "lung", "liver", "kidney", "muscle"] if o in TRANSFER_ORGANS]


def _clock_gene_weights(acc_genes: pd.DataFrame, weights: pd.Series) -> pd.Series:
    gene_of = acc_genes.drop_duplicates("accession").set_index("accession")["gene"]
    pooled: dict[str, float] = {}
    for acc, weight in weights.items():
        acc = str(acc)
        gene = str(gene_of[acc]) if acc in gene_of.index else ""
        gene = _normalize_gene(gene)
        if not gene:
            continue
        pooled[gene] = pooled.get(gene, 0.0) + float(weight)
    return pd.Series(pooled, dtype=float)


def _prepare_rna(
    data_dir: Path,
    organ: str,
    metadata: pd.DataFrame,
) -> dict:
    values, samples = load_whole_tissue(data_dir, organ, metadata)
    del values
    rna = load_rna_matrix(data_dir, organ)
    samples = _align_samples(samples, rna)
    if samples.empty:
        raise ValueError(f"No RNA columns matched TMT samples for {organ}")
    rna_cols = samples["rna_col"].tolist()
    rna_log = np.log2(rna.clip(lower=0.0).reindex(columns=rna_cols) + 1.0)
    rna_log.columns = samples["sample_id"].tolist()
    acc_genes = load_accession_genes(data_dir, organ, metadata)
    return {
        "organ": organ,
        "samples": samples,
        "ages": samples["age_months"].to_numpy(dtype=float),
        "sample_ids": samples["sample_id"].tolist(),
        "rna_log": rna_log,
        "acc_genes": acc_genes,
    }


def score_clock_on_rna(clock_w: pd.Series, rna_bundle: dict) -> dict[str, float]:
    rna_log = rna_bundle["rna_log"]
    ages = rna_bundle["ages"]
    matched = [g for g in clock_w.index.astype(str) if g in rna_log.index]
    n_clock = int((clock_w.astype(float).abs() > 0).sum())
    n_matched = int(len(matched))
    if n_matched < MIN_MATCHED:
        return {
            "n_clock_genes": n_clock,
            "n_matched": n_matched,
            "n_samples": int(len(ages)),
            "spearman_vs_age": np.nan,
            "p_parametric": np.nan,
            "p_permutation": np.nan,
            "pearson_vs_age": np.nan,
        }
    w_use = clock_w.reindex(matched).astype(float)
    block = rna_log.reindex(index=matched, columns=rna_bundle["sample_ids"])
    scores = _score_matched(block, w_use).reindex(rna_bundle["sample_ids"]).to_numpy(dtype=float)
    r, p, pp = permutation_spearman(scores, ages)
    return {
        "n_clock_genes": n_clock,
        "n_matched": n_matched,
        "n_samples": int(len(ages)),
        "spearman_vs_age": r,
        "p_parametric": p,
        "p_permutation": pp,
        "pearson_vs_age": _pearson(scores, ages),
    }


def write_rna_cross_organ(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "rna_cross_organ"
    dest.mkdir(parents=True, exist_ok=True)
    metadata = load_takasugi_metadata(data_dir)
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    organs = [o for o in PANEL_ORGANS if o in RNA_FILES]
    print("  Loading RNA matrices ...", flush=True)
    bundles = {organ: _prepare_rna(data_dir, organ, metadata) for organ in organs}
    clocks: dict[str, pd.Series] = {}
    for organ in organs:
        weights = _frozen_weights(stab_dir, coef_dir, organ)["stable"]
        clocks[organ] = _clock_gene_weights(bundles[organ]["acc_genes"], weights)

    rows = []
    for clock_organ in organs:
        w = clocks[clock_organ]
        for rna_organ in organs:
            stats = score_clock_on_rna(w, bundles[rna_organ])
            same = clock_organ == rna_organ
            row = {
                "clock_organ": clock_organ,
                "rna_organ": rna_organ,
                "same_organ": same,
                **stats,
            }
            rows.append(row)
            rho = stats["spearman_vs_age"]
            rho_s = f"{rho:.3f}" if np.isfinite(rho) else "NA"
            print(
                f"    {clock_organ:7s} → {rna_organ:7s} RNA  "
                f"ρ={rho_s}  genes={stats['n_matched']}/{stats['n_clock_genes']}",
                flush=True,
            )
    metrics = pd.DataFrame(rows)
    diag = metrics.loc[metrics["same_organ"], "spearman_vs_age"].to_numpy(dtype=float)
    off = metrics.loc[~metrics["same_organ"], "spearman_vs_age"].to_numpy(dtype=float)
    winners = []
    for rna_organ in organs:
        sub = metrics.loc[metrics["rna_organ"] == rna_organ]
        matched_rho = float(sub.loc[sub["clock_organ"] == rna_organ, "spearman_vs_age"].iloc[0])
        best_rho = float(sub["spearman_vs_age"].max())
        bests = sub.loc[np.isclose(sub["spearman_vs_age"].to_numpy(dtype=float), best_rho)].copy()
        best_clocks = bests["clock_organ"].tolist()
        winners.append(
            {
                "rna_organ": rna_organ,
                "best_clock": ",".join(best_clocks),
                "best_is_matched": rna_organ in best_clocks,
                "matched_is_unique_best": best_clocks == [rna_organ],
                "matched_rho": matched_rho,
                "best_rho": best_rho,
            }
        )
    n_sig_cross = int(
        (
            (~metrics["same_organ"])
            & metrics["p_permutation"].le(0.05)
            & metrics["spearman_vs_age"].gt(0)
        ).sum()
    )
    summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(metrics)),
                "n_same": int(metrics["same_organ"].sum()),
                "n_cross": int((~metrics["same_organ"]).sum()),
                "same_mean_spearman": float(np.nanmean(diag)),
                "cross_mean_spearman": float(np.nanmean(off)),
                "n_cross_positive_p05": n_sig_cross,
                "n_rna_organs_matched_clock_tied_or_best": int(sum(w["best_is_matched"] for w in winners)),
                "n_rna_organs_matched_clock_unique_best": int(sum(w["matched_is_unique_best"] for w in winners)),
                "n_rna_organs": int(len(winners)),
            }
        ]
    )
    win_frame = pd.DataFrame(winners)
    metrics.to_csv(dest / "metrics.csv", index=False)
    summary.to_csv(dest / "summary.csv", index=False)
    win_frame.to_csv(dest / "winners.csv", index=False)
    print(
        f"  same-organ mean ρ={float(np.nanmean(diag)):.3f}  "
        f"cross-organ mean ρ={float(np.nanmean(off)):.3f}  "
        f"matched unique best {int(sum(w['matched_is_unique_best'] for w in winners))}/{len(winners)}; "
        f"tied or best {int(sum(w['best_is_matched'] for w in winners))}/{len(winners)}; "
        f"cross-organ ρ>0 p≤0.05: {n_sig_cross}/{int((~metrics['same_organ']).sum())}",
        flush=True,
    )
    return {"metrics": metrics, "summary": summary, "winners": win_frame}
