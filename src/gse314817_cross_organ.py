"""Cross-organ RNA projection of frozen Scifo clocks on GSE314817. No refit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ageaccel import permutation_spearman
from .gse314817_transfer import (
    PANEL_ORGANS,
    _default_data_dir,
    _subset_samples,
    load_gse314817_matrix,
    load_gse314817_metadata,
)
from .rna_cross_organ import score_clock_on_rna
from .takasugi_transfer import _frozen_weights, load_takasugi_metadata
from .tms_transfer import _gene_weights_for_organ, _lookup_index


def _map_weights(w_gene: pd.Series, lookup: dict[str, str]) -> pd.Series:
    mapped: dict[str, float] = {}
    for gene, w in w_gene.items():
        hit = lookup.get(str(gene)) or lookup.get(str(gene).lower())
        if hit:
            mapped[hit] = mapped.get(hit, 0.0) + float(w)
    return pd.Series(mapped, dtype=float)


def load_gse314817_bundles(root: Path, out_dir: Path, data_dir: Path | None = None) -> dict:
    data_dir = data_dir or _default_data_dir(root)
    meta = load_gse314817_metadata(data_dir)
    rna = load_gse314817_matrix(data_dir)
    lookup = _lookup_index(rna.index)
    takasugi_dir = root / "Dataset" / "Takasugi"
    takasugi_meta = load_takasugi_metadata(takasugi_dir)
    stable_genes = pd.read_csv(out_dir / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    stab_dir = out_dir / "clocks" / "stability"
    coef_dir = out_dir / "clocks" / "coefficients"
    organs = [o for o in PANEL_ORGANS if (stab_dir / f"{o}.csv").exists() and (meta["organ"] == o).any()]
    clocks = {}
    rna_log = {}
    sample_ids = {}
    ages = {}
    tissues = {}
    for organ in organs:
        weights = _frozen_weights(stab_dir, coef_dir, organ)["stable"]
        w_gene = _gene_weights_for_organ(organ, weights, takasugi_dir, takasugi_meta, stable_genes)
        clocks[organ] = _map_weights(w_gene, lookup)
        samples = _subset_samples(meta, organ, "male")
        cols = [c for c in samples["sample_id"] if c in rna.columns]
        samples = samples.loc[samples["sample_id"].isin(cols)].copy()
        ids = samples["sample_id"].tolist()
        sample_ids[organ] = ids
        ages[organ] = samples["age_months"].to_numpy(dtype=float)
        tissues[organ] = str(samples["tissue"].iloc[0]) if not samples.empty else organ
        rna_log[organ] = np.log2(rna.reindex(columns=ids).clip(lower=0.0) + 1.0)
    return {
        "organs": organs,
        "clocks": clocks,
        "rna_log": rna_log,
        "sample_ids": sample_ids,
        "ages": ages,
        "tissues": tissues,
        "lookup": lookup,
        "rna": rna,
        "meta": meta,
    }


def _bundle(loaded: dict, organ: str) -> dict:
    return {
        "organ": organ,
        "ages": loaded["ages"][organ],
        "sample_ids": loaded["sample_ids"][organ],
        "rna_log": loaded["rna_log"][organ],
    }


def write_gse314817_cross_organ(root: Path, out_dir: Path, loaded: dict | None = None) -> dict[str, pd.DataFrame]:
    dest = out_dir / "gse314817_cross_organ"
    dest.mkdir(parents=True, exist_ok=True)
    if loaded is None:
        print("  Loading GSE314817 for cross-organ RNA ...", flush=True)
        loaded = load_gse314817_bundles(root, out_dir)
    organs = loaded["organs"]
    rows = []
    for clock_organ in organs:
        w = loaded["clocks"][clock_organ]
        for rna_organ in organs:
            stats = score_clock_on_rna(w, _bundle(loaded, rna_organ))
            same = clock_organ == rna_organ
            row = {
                "clock_organ": clock_organ,
                "clock_tissue": loaded["tissues"][clock_organ],
                "rna_organ": rna_organ,
                "rna_tissue": loaded["tissues"][rna_organ],
                "same_organ": same,
                **stats,
            }
            rows.append(row)
            rho = stats["spearman_vs_age"]
            rho_s = f"{rho:.3f}" if np.isfinite(rho) else "NA"
            print(
                f"    {clock_organ:7s} → {rna_organ:7s} RNA  ρ={rho_s}  "
                f"genes={stats['n_matched']}/{stats['n_clock_genes']}",
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
        bests = sub.loc[np.isclose(sub["spearman_vs_age"].to_numpy(dtype=float), best_rho)]
        best_clocks = bests["clock_organ"].tolist()
        winners.append(
            {
                "rna_organ": rna_organ,
                "rna_tissue": loaded["tissues"][rna_organ],
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
        f"  GSE314817 same-organ mean ρ={float(np.nanmean(diag)):.3f}  "
        f"cross-organ mean ρ={float(np.nanmean(off)):.3f}  "
        f"matched unique best {int(sum(w['matched_is_unique_best'] for w in winners))}/{len(winners)}; "
        f"cross-organ ρ>0 p≤0.05: {n_sig_cross}/{int((~metrics['same_organ']).sum())}",
        flush=True,
    )
    return {"metrics": metrics, "summary": summary, "winners": win_frame, "loaded": loaded}
