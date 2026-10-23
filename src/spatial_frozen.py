"""Frozen Scifo protein-clock weights on Visium sections, niches, and spots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .ageaccel import permutation_spearman
from .rna_portability import N_NULL, _match_values, _p_from_null, _score_rho_np, _score_variants
from .spatial_io import SECTION_ORGANS, load_section
from .spatial_clocks import BRAIN_LEVELS, SCIFO_ORGAN, _spearman, log_cpm
from .takasugi_rna import MIN_MATCHED, _score_matched

RNG = np.random.default_rng(17)
N_HVG = 2_000
N_PCS = 30
MIN_NICHE_SPOTS = 30
MIN_NICHE_MICE = 0.8


def _weights_for(scifo: pd.DataFrame, organ: str, gene_lookup: dict[str, str]) -> tuple[pd.Series, int]:
    reference = scifo.loc[scifo["organ"] == SCIFO_ORGAN[organ]]
    pooled: dict[str, float] = {}
    for gene, weight in zip(reference["gene"].astype(str), reference["mean_coef"].astype(float)):
        hit = gene_lookup.get(gene) or gene_lookup.get(gene.lower())
        if hit:
            pooled[hit] = pooled.get(hit, 0.0) + float(weight)
    return pd.Series(pooled, dtype=float), int(reference["gene"].nunique())


def _lookup(genes: pd.Index) -> dict[str, str]:
    out = {}
    for gene in genes.astype(str):
        out[gene] = gene
        out.setdefault(gene.lower(), gene)
    return out


def write_frozen_sections(out_dir: Path, results_root: Path) -> pd.DataFrame:
    """Score each mouse's section with frozen protein weights. No refit."""
    dest = out_dir / "frozen"
    dest.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(out_dir / "samples.csv")
    scifo = pd.read_csv(results_root / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    score_rows = []
    metric_rows = []
    null_rows = []
    organs = ["Brain", *SECTION_ORGANS]
    brain_levels = {}
    for organ in BRAIN_LEVELS:
        counts = pd.read_csv(out_dir / "pseudobulk" / f"{organ}.csv", index_col=0)
        counts.index = counts.index.astype(str)
        counts.columns = counts.columns.astype(str)
        brain_levels[organ] = counts
    from .spatial_clocks import pooled_brain

    brain_log, _fraction = pooled_brain(brain_levels)
    matrices: dict[str, pd.DataFrame] = {"Brain": brain_log}
    sample_organs = {"Brain": samples.loc[samples["organ"].isin(BRAIN_LEVELS)].drop_duplicates("mouse_id")}
    for organ in SECTION_ORGANS:
        counts = brain_levels.get(organ)
        if counts is None:
            counts = pd.read_csv(out_dir / "pseudobulk" / f"{organ}.csv", index_col=0)
            counts.index = counts.index.astype(str)
            counts.columns = counts.columns.astype(str)
        matrices[organ] = log_cpm(counts)
        sample_organs[organ] = samples.loc[samples["organ"] == organ]
    for organ in organs:
        logged = matrices[organ]
        weights, n_clock = _weights_for(scifo, organ, _lookup(logged.index))
        group = sample_organs[organ].sort_values("mouse_id")
        columns = [str(mouse) for mouse in group["mouse_id"] if str(mouse) in logged.columns]
        group = group.loc[group["mouse_id"].astype(str).isin(columns)]
        ages = group["age_months"].to_numpy(dtype=float)
        block = logged.reindex(index=weights.index, columns=columns)
        if len(weights) < MIN_MATCHED or len(columns) < 5:
            metric_rows.append(
                {
                    "organ": organ,
                    "n_clock_genes": n_clock,
                    "n_matched": int(len(weights)),
                    "n_mice": int(len(columns)),
                    "note": "too_few",
                }
            )
            continue
        variants = _score_variants(block, weights)
        exact = variants["exact_weight"].reindex(columns).to_numpy(dtype=float)
        rho, _p_param, p_perm = permutation_spearman(exact, ages)
        for name, series in variants.items():
            vals = series.reindex(columns).to_numpy(dtype=float)
            metric_rows.append(
                {
                    "organ": organ,
                    "score": name,
                    "spearman_vs_age": _spearman(vals, ages),
                    "p_permutation": p_perm if name == "exact_weight" else np.nan,
                    "n_clock_genes": n_clock,
                    "n_matched": int(len(weights)),
                    "n_mice": int(len(columns)),
                }
            )
        scored = group[["mouse_id", "age_weeks", "age_months"]].copy()
        scored["organ"] = organ
        scored["score"] = exact
        scored["n_matched"] = int(len(weights))
        score_rows.append(scored)
        print(
            f"  frozen {organ}: genes {len(weights)}/{n_clock}  n={len(columns)}  "
            f"rho={rho:.3f}  p={p_perm:.4g}",
            flush=True,
        )
        full = logged.reindex(columns=columns)
        detected = full.index[(full.to_numpy(dtype=float) > 0).mean(axis=1) > 0.5]
        detected = np.array([gene for gene in detected if gene not in set(weights.index)])
        abundance = full.mean(axis=1)
        clock_abundance = abundance.reindex(weights.index).to_numpy(dtype=float)
        gene_ids = abundance.index.astype(str).to_numpy()
        pool_mask = abundance.index.isin(detected)
        mat = full.to_numpy(dtype=float)
        index_of = {gene: i for i, gene in enumerate(full.index.astype(str))}
        w_vals = weights.to_numpy(dtype=float)
        draws = np.full(N_NULL, np.nan)
        for i in range(N_NULL):
            picked = _match_values(clock_abundance, gene_ids[pool_mask], abundance.to_numpy()[pool_mask], RNG)
            if not picked:
                continue
            paired = [(index_of[gene], weight) for gene, weight in zip(picked, w_vals) if gene in index_of]
            if len(paired) < MIN_MATCHED:
                continue
            row_idx = np.asarray([row for row, _weight in paired], dtype=int)
            draws[i] = _score_rho_np(mat[row_idx], np.asarray([weight for _row, weight in paired]), ages)
        stats = _p_from_null(float(rho), draws)
        null_rows.append({"organ": organ, "score": "exact_weight", "null": "expression_matched", "observed": rho, **stats})
        print(f"    expression-matched null p={stats['p_null']:.4g}  null mean={stats['null_mean']:.3f}", flush=True)
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    nulls = pd.DataFrame(null_rows)
    scores.to_csv(dest / "scores.csv", index=False)
    metrics.to_csv(dest / "metrics.csv", index=False)
    nulls.to_csv(dest / "nulls.csv", index=False)
    return scores


def _gene_variance(counts: sparse.spmatrix) -> np.ndarray:
    """Variance of log1p counts per 10,000, computed in gene chunks."""
    library = np.asarray(counts.sum(axis=0)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    variance = np.zeros(counts.shape[0], dtype=float)
    for start in range(0, counts.shape[0], 2000):
        block = np.log1p(counts[start : start + 2000].multiply(scale).toarray())
        variance[start : start + block.shape[0]] = block.var(axis=1)
    return variance


def _hvg_matrix(counts: sparse.spmatrix, genes: np.ndarray) -> np.ndarray:
    library = np.asarray(counts.sum(axis=0)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(counts[genes].multiply(scale).toarray())


def _cluster_organ(sections: list[dict]) -> tuple[np.ndarray, list[int], dict[int, str]]:
    """Cluster spots after removing each section's gene mean."""
    total_spots = sum(section["counts"].shape[1] for section in sections)
    variance = np.zeros(sections[0]["counts"].shape[0], dtype=float)
    for section in sections:
        variance += _gene_variance(section["counts"]) * section["counts"].shape[1]
    variance /= total_spots
    chosen = np.argsort(variance)[-N_HVG:]
    centered = []
    offsets = []
    cursor = 0
    for section in sections:
        block = _hvg_matrix(section["counts"], chosen)
        centered.append(block - block.mean(axis=1, keepdims=True))
        offsets.append(cursor)
        cursor += block.shape[1]
    matrix = np.concatenate(centered, axis=1).T
    n_components = min(N_PCS, matrix.shape[0] - 1, matrix.shape[1])
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    n_clusters = 10 if str(sections[0]["organ"]).startswith("Brain") else 8
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=10, batch_size=2048).fit_predict(pcs)
    raw = [_hvg_matrix(section["counts"], chosen) for section in sections]
    expression = np.concatenate(raw, axis=1)
    symbols = sections[0]["genes"][chosen]
    markers = {}
    for cluster in range(n_clusters):
        inside = labels == cluster
        if int(inside.sum()) < MIN_NICHE_SPOTS or int((~inside).sum()) == 0:
            markers[cluster] = ""
            continue
        delta = expression[:, inside].mean(axis=1) - expression[:, ~inside].mean(axis=1)
        markers[cluster] = ",".join(symbols[np.argsort(delta)[-3:]][::-1])
    return labels, offsets, markers


def _representative(group: pd.DataFrame, spot_counts: dict[int, int]) -> list[int]:
    chosen = []
    for _age, sub in group.groupby("age_weeks"):
        mice = sub["mouse_id"].astype(int).tolist()
        counts = np.array([spot_counts[mouse] for mouse in mice])
        chosen.append(mice[int(np.argmin(np.abs(counts - np.median(counts))))])
    return chosen


def write_niches(out_dir: Path, results_root: Path) -> None:
    dest = out_dir / "niches"
    map_dir = out_dir / "spot_maps"
    dest.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(out_dir / "samples.csv")
    scifo = pd.read_csv(results_root / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    assignments = []
    score_rows = []
    composition_rows = []
    metric_rows = []
    for organ in SECTION_ORGANS:
        group = samples.loc[samples["organ"] == organ].sort_values(["age_weeks", "mouse_id"])
        print(f"  niches {organ}: {len(group)} sections", flush=True)
        sections = []
        for row in group.itertuples(index=False):
            loaded = load_section(Path(row.h5_path), Path(row.positions_path))
            loaded["organ"] = organ
            loaded["mouse_id"] = int(row.mouse_id)
            loaded["age_weeks"] = int(row.age_weeks)
            loaded["age_months"] = float(row.age_months)
            loaded["gsm"] = row.gsm
            sections.append(loaded)
        labels, offsets, markers = _cluster_organ(sections)
        spot_counts = {}
        for section, start in zip(sections, offsets):
            stop = start + section["counts"].shape[1]
            section["labels"] = labels[start:stop]
            spot_counts[section["mouse_id"]] = int(section["counts"].shape[1])
            assignments.append(
                pd.DataFrame(
                    {
                        "gsm": section["gsm"],
                        "organ": organ,
                        "mouse_id": section["mouse_id"],
                        "age_weeks": section["age_weeks"],
                        "barcode": section["barcodes"],
                        "niche": section["labels"],
                        "pxl_col": section["pxl_col"],
                        "pxl_row": section["pxl_row"],
                    }
                )
            )
        weights, _n_clock = _weights_for(scifo, organ, _lookup(pd.Index(sections[0]["genes"])))
        n_mice = group["mouse_id"].nunique()
        min_mice = int(np.ceil(MIN_NICHE_MICE * n_mice))
        gene_index = {gene: i for i, gene in enumerate(sections[0]["genes"])}
        weight_rows = [gene_index[gene] for gene in weights.index if gene in gene_index]
        weight_genes = [gene for gene in weights.index if gene in gene_index]
        use_weights = weights.reindex(weight_genes)
        niches = sorted(set(labels.tolist()))
        for niche in niches:
            present = []
            profiles = []
            mice = []
            ages = []
            weeks = []
            fractions = []
            for section in sections:
                mask = section["labels"] == niche
                n_spots = int(mask.sum())
                fractions.append(n_spots / section["counts"].shape[1])
                if n_spots < MIN_NICHE_SPOTS:
                    continue
                totals = np.asarray(section["counts"][:, mask].sum(axis=1)).ravel()
                profiles.append(totals)
                present.append(section)
                mice.append(section["mouse_id"])
                ages.append(section["age_months"])
                weeks.append(section["age_weeks"])
            if len(set(mice)) < min_mice or len(use_weights) < MIN_MATCHED:
                continue
            counts = pd.DataFrame(np.column_stack(profiles), index=sections[0]["genes"], columns=[str(m) for m in mice])
            logged = log_cpm(counts)
            block = logged.reindex(index=use_weights.index)
            scores = _score_matched(block, use_weights).to_numpy(dtype=float)
            age_array = np.asarray(ages, dtype=float)
            week_array = np.asarray(weeks, dtype=int)
            rho, _param, p_score = permutation_spearman(scores, age_array)
            frac = np.asarray(fractions, dtype=float)
            frac_ages = np.array([section["age_months"] for section in sections], dtype=float)
            rho_frac, _param_f, p_frac = permutation_spearman(frac, frac_ages)
            label = f"{niche}:{markers.get(niche, '')}"
            metric_rows.append(
                {
                    "organ": organ,
                    "niche": int(niche),
                    "markers": markers.get(niche, ""),
                    "n_mice": int(len(set(mice))),
                    "n_sections": int(len(mice)),
                    "spearman_score_age": rho,
                    "p_score": p_score,
                    "spearman_fraction_age": rho_frac,
                    "p_fraction": p_frac,
                    "mean_score_17": float(np.nanmean(scores[week_array == 17])) if np.any(week_array == 17) else np.nan,
                    "mean_score_73": float(np.nanmean(scores[week_array == 73])) if np.any(week_array == 73) else np.nan,
                    "mean_score_112": float(np.nanmean(scores[week_array == 112])) if np.any(week_array == 112) else np.nan,
                }
            )
            for mouse, age, week, score in zip(mice, ages, weeks, scores):
                score_rows.append(
                    {
                        "organ": organ,
                        "niche": int(niche),
                        "markers": markers.get(niche, ""),
                        "mouse_id": int(mouse),
                        "age_weeks": int(week),
                        "age_months": float(age),
                        "score": float(score),
                    }
                )
            for section, fraction in zip(sections, fractions):
                composition_rows.append(
                    {
                        "organ": organ,
                        "niche": int(niche),
                        "mouse_id": section["mouse_id"],
                        "age_weeks": section["age_weeks"],
                        "age_months": section["age_months"],
                        "fraction": float(fraction),
                    }
                )
            print(f"    {label}  mice={len(set(mice))}  score rho={rho:.3f}  fraction rho={rho_frac:.3f}", flush=True)
        chosen = set(_representative(group, spot_counts))
        map_rows = []
        if weight_rows:
            blocks = []
            for section in sections:
                library = np.asarray(section["counts"].sum(axis=0)).ravel()
                scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
                blocks.append(np.log1p(section["counts"][weight_rows].multiply(scale).toarray()))
            stacked = np.concatenate(blocks, axis=1)
            center = stacked.mean(axis=1, keepdims=True)
            spread = np.maximum(stacked.std(axis=1, ddof=0, keepdims=True), 1e-12)
            scaled = (stacked - center) / spread
            cursor = 0
            gene_weights = use_weights.to_numpy(dtype=float)
            for section in sections:
                n_spots = section["counts"].shape[1]
                if section["mouse_id"] in chosen:
                    values = gene_weights @ np.nan_to_num(scaled[:, cursor : cursor + n_spots])
                    map_rows.append(
                        pd.DataFrame(
                            {
                                "organ": organ,
                                "mouse_id": section["mouse_id"],
                                "age_weeks": section["age_weeks"],
                                "age_months": section["age_months"],
                                "pxl_col": section["pxl_col"],
                                "pxl_row": section["pxl_row"],
                                "score": values,
                            }
                        )
                    )
                cursor += n_spots
        if map_rows:
            pd.concat(map_rows, ignore_index=True).to_csv(map_dir / f"{organ}.csv", index=False)
    if assignments:
        pd.concat(assignments, ignore_index=True).to_csv(dest / "assignments.csv", index=False)
    pd.DataFrame(score_rows).to_csv(dest / "scores.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(dest / "composition.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(dest / "metrics.csv", index=False)
