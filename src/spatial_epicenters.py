"""Search Visium sections for spots that act as sources of the protein-clock score.

The score at every spot is the frozen protein-clock score with the spot's niche mean
and sequencing depth removed, so a high-scoring anatomical region does not count as a
source by itself. The mouse is the unit in every age comparison.

Source scan: for curated spot types and 200 random gene-marked spot sets, the
within-section Spearman correlation between that score and distance to the nearest
marked spot. A curated type is a candidate source when its correlation is stronger, or
strengthens more with age, than the random sets.

Blind epicenters: clock genes are split into two halves. Local maxima of the smoothed
score from one half are candidate epicenters, and the score from the other half measures
the gradient around them against niche-matched random centres. The halves are then
swapped, so an epicenter is never confirmed by its own counting noise.
"""

from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .spatial_followup import _um_per_pixel
from .spatial_frozen import _weights_for
from .spatial_io import SECTION_ORGANS, load_section

from .paths import project_root

ROOT = project_root()
OUT = ROOT / "results" / "spatial_aging"
DEST = OUT / "epicenters"
N_PERM = 5_000
N_RANDOM = 200
N_NULL = 50
SIGMA_UM = 100.0
PEAK_RADIUS_UM = 300.0
PROFILE_UM = 600.0
SOURCE_RADIUS_UM = 750.0
RING_EDGES = np.array([50.0, 150.0, 250.0, 350.0, 450.0, PROFILE_UM])
RING_LABELS = ["0", "100", "200", "300", "400", "525"]
MIN_NEIGHBOURS = 24
BIN_EDGES = np.array([0.0, 150.0, 250.0, 350.0, 450.0, 600.0, SOURCE_RADIUS_UM])
BIN_LABELS = ["100", "200", "300", "400", "525", "675"]
BRAIN = ("Brain1", "Brain2", "Brain3", "Brain4", "Brain5")
MAP_MODULES = ("mhc_i", "microglia_complement", "immune_panel", "myelin_anatomy_control")

COMPOSITION = {
    "oligodendrocyte": ("Plp1", "Mbp", "Mobp", "Mag", "Cnp", "Cldn11", "Mog", "Ermn"),
    "neuron": ("Snap25", "Syt1", "Rbfox3", "Stmn2", "Nrgn"),
    "astrocyte": ("Aqp4", "Slc1a2", "Gja1", "Aldoc", "Slc1a3"),
}

SOURCES = {
    "immune_panel": ("Ptprc", "Cd3e", "Cd79a", "C1qa", "Lcn2"),
    "lymphocyte": ("Ptprc", "Cd3e", "Cd3g", "Cd8a", "Cd79a", "Ms4a1"),
    "microglia_complement": ("C1qa", "C1qb", "C1qc", "Tyrobp", "Trem2"),
    "mhc_ii": ("Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1"),
    "interferon": ("Ifit1", "Ifit3", "Isg15", "Irf7", "Rtp4", "Bst2", "Ifi27l2a"),
    "mhc_i": ("B2m", "H2-K1", "H2-D1", "H2-Q7"),
    "c4b_reactive_astro": ("C4b", "Serpina3n", "Gfap", "Cd44", "Vim"),
    "endothelium": ("Cldn5", "Flt1", "Pecam1", "Ly6c1", "Slco1a4"),
    "perivascular_macrophage": ("Mrc1", "Lyve1", "Pf4", "Cd163", "F13a1"),
    "choroid_plexus": ("Ttr", "Kl", "Folr1", "Clic6"),
    "senescence": ("Cdkn2a", "Cdkn1a", "Glb1", "Serpine1"),
    "acute_phase": ("Lcn2", "Lrg1", "Timp1", "Socs3"),
    "myelin_anatomy_control": ("Plp1", "Mbp", "Mobp"),
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _perm_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan"), float("nan")
    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan")
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    observed = float(np.mean(rx * ry))
    order = np.argsort(rng.random((N_PERM, len(ry))), axis=1)
    null = np.mean(rx[None, :] * ry[order], axis=1)
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (N_PERM + 1)
    return observed, float(p)


def _sign_flip(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 4:
        return float("nan")
    observed = abs(values.mean())
    signs = np.array(list(product((-1.0, 1.0), repeat=n)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def _score(counts: sparse.csr_matrix, index: dict[str, int], library: np.ndarray, weights: pd.Series) -> np.ndarray:
    rows, coef = [], []
    for gene, weight in weights.items():
        hit = index.get(str(gene).lower())
        if hit is not None and np.isfinite(weight):
            rows.append(hit)
            coef.append(float(weight))
    if len(rows) < 3:
        return np.full(counts.shape[1], np.nan)
    scale = np.divide(1e6, library, out=np.zeros_like(library), where=library > 0)
    logged = np.log2(counts[rows].toarray() * scale[None, :] + 1.0)
    mean = logged.mean(axis=1, keepdims=True)
    std = logged.std(axis=1, keepdims=True)
    std[std == 0] = np.nan
    z = (logged - mean) / std
    use = np.isfinite(z).all(axis=1)
    return z[use].T @ np.asarray(coef)[use]


def _residual(score: np.ndarray, design: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ok = valid & np.isfinite(score)
    beta = np.linalg.lstsq(design[ok], score[ok], rcond=None)[0]
    out = np.full(score.shape, np.nan)
    out[ok] = score[ok] - design[ok] @ beta
    out[ok] /= out[ok].std()
    return out


def _module(counts: sparse.csr_matrix, index: dict[str, int], library: np.ndarray, genes) -> np.ndarray | None:
    rows = [index[g.lower()] for g in genes if g.lower() in index]
    if len(rows) < 2:
        return None
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(counts[rows].toarray() * scale[None, :]).mean(axis=0)


def _source_rho(module: np.ndarray, residual: np.ndarray, score: np.ndarray, points: np.ndarray, valid: np.ndarray) -> dict:
    out = {"rho_adj": np.nan, "rho_raw": np.nan, "n_source": 0, **{f"bin_{lab}": np.nan for lab in BIN_LABELS}}
    finite = valid & np.isfinite(module)
    if finite.sum() < 200:
        return out
    cutoff = np.quantile(module[finite], 0.90)
    source = finite & (module >= cutoff) & (module > 0)
    out["n_source"] = int(source.sum())
    if out["n_source"] < 30 or out["n_source"] > 0.4 * finite.sum():
        return out
    other = finite & ~source
    distance, _ = cKDTree(points[source]).query(points[other])
    near = distance <= SOURCE_RADIUS_UM
    if near.sum() < 80:
        return out
    out["rho_adj"] = _spearman(residual[other][near], distance[near])
    out["rho_raw"] = _spearman(score[other][near], distance[near])
    which = np.searchsorted(BIN_EDGES, distance, side="left")
    values = residual[other]
    for k, lab in enumerate(BIN_LABELS, start=1):
        sel = which == k
        if sel.sum() >= 10:
            out[f"bin_{lab}"] = float(np.nanmean(values[sel]))
    return out


def _neighbour_graph(points: np.ndarray, radius: float) -> sparse.csr_matrix:
    tree = cKDTree(points)
    pairs = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    keep = pairs.row != pairs.col
    return sparse.csr_matrix((pairs.data[keep], (pairs.row[keep], pairs.col[keep])), shape=(len(points), len(points)))


def _smooth(values: np.ndarray, graph: sparse.csr_matrix) -> np.ndarray:
    weights = graph.copy()
    weights.data = np.exp(-(weights.data**2) / (2 * SIGMA_UM**2))
    weights.data[graph.data > 3 * SIGMA_UM] = 0.0
    mask = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    out = (weights @ filled + filled) / (weights @ mask + mask)
    out[mask == 0] = np.nan
    return out


def _peaks(smoothed: np.ndarray, graph: sparse.csr_matrix, n_close: np.ndarray) -> np.ndarray:
    close = graph.copy()
    close.data = (close.data <= PEAK_RADIUS_UM).astype(float)
    close.eliminate_zeros()
    filled = np.where(np.isfinite(smoothed), smoothed, -np.inf)
    neighbour_max = np.full(len(smoothed), -np.inf)
    for i in range(len(smoothed)):
        start, end = close.indptr[i], close.indptr[i + 1]
        if end > start:
            neighbour_max[i] = filled[close.indices[start:end]].max()
    threshold = np.nanquantile(smoothed, 0.95)
    return np.where(np.isfinite(smoothed) & (filled >= neighbour_max) & (smoothed >= threshold) & (n_close >= MIN_NEIGHBOURS))[0]


def _ring_profile(centres: np.ndarray, values: np.ndarray, graph: sparse.csr_matrix) -> np.ndarray:
    rows = []
    for c in centres:
        start, end = graph.indptr[c], graph.indptr[c + 1]
        idx = graph.indices[start:end]
        dist = graph.data[start:end]
        ring = np.searchsorted(RING_EDGES, dist, side="left")
        profile = [values[c]]
        for k in range(1, len(RING_EDGES)):
            sel = ring == k
            profile.append(np.nanmean(values[idx[sel]]) if sel.any() else np.nan)
        rows.append(profile)
    return np.nanmean(np.asarray(rows, dtype=float), axis=0) if rows else np.full(len(RING_EDGES), np.nan)


def _null_centres(centres, niche, n_close, valid, points, rng) -> list[np.ndarray]:
    far = np.ones(len(niche), dtype=bool)
    if len(centres):
        distance, _ = cKDTree(points[centres]).query(points)
        far = distance > 150.0
    draws = []
    pools = []
    for c in centres:
        pool = np.where(valid & far & (niche == niche[c]) & (np.abs(n_close - n_close[c]) <= 4) & (n_close >= MIN_NEIGHBOURS))[0]
        pools.append(pool)
    for _ in range(N_NULL):
        draws.append(np.array([rng.choice(pool) for pool in pools if len(pool)], dtype=int))
    return draws


def _strength(profile: np.ndarray) -> tuple[float, float]:
    inner = np.nanmean(profile[:2])
    outer = np.nanmean(profile[4:])
    return float(inner - outer), float(inner)


def _sections(organ: str, samples: pd.DataFrame, assignments: pd.DataFrame, weights_table: pd.DataFrame, rng):
    group = samples.loc[samples["organ"] == organ]
    probe = load_section(Path(group.iloc[0].h5_path), Path(group.iloc[0].positions_path))
    genes = probe["genes"]
    index = {gene.lower(): i for i, gene in enumerate(genes)}
    weights, _n = _weights_for(weights_table, organ, {gene.lower(): gene for gene in genes})
    order = rng.permutation(len(weights))
    halves = (weights.iloc[np.sort(order[: len(order) // 2])], weights.iloc[np.sort(order[len(order) // 2 :])])
    clock = {str(g).lower() for g in weights.index}
    detected = np.asarray((probe["counts"] > 0).mean(axis=1)).ravel()
    pool = np.array(
        [
            i
            for i, g in enumerate(genes)
            if 0.10 <= detected[i] <= 0.70 and not g.lower().startswith(("mt-", "rpl", "rps")) and g.lower() not in clock
        ]
    )
    random_sets = {f"random_{k:03d}": tuple(genes[rng.choice(pool, 5, replace=False)]) for k in range(N_RANDOM)}
    return group, genes, index, weights, halves, clock, random_sets


def _blind_section(counts, index, library, halves, design, valid, niche, points, rng) -> dict:
    graph = _neighbour_graph(points, PROFILE_UM)
    within = graph.copy()
    within.data = (within.data <= PEAK_RADIUS_UM).astype(float)
    n_close = np.asarray(within.sum(axis=1)).ravel()
    res_halves = [_residual(_score(counts, index, library, half), design, valid) for half in halves]
    smooth_halves = [_smooth(r, graph) for r in res_halves]
    profiles_epi, profiles_null, centres_all, null_all = [], [], [], []
    for detect, measure in ((0, 1), (1, 0)):
        centres = _peaks(smooth_halves[detect], graph, n_close)
        draws = _null_centres(centres, niche, n_close, valid, points, rng)
        profiles_epi.append(_ring_profile(centres, res_halves[measure], graph))
        profiles_null.append(np.nanmean([_ring_profile(d, res_halves[measure], graph) for d in draws], axis=0))
        centres_all.append(centres)
        null_all.extend(draws[:10])
    epi = np.nanmean(profiles_epi, axis=0)
    null = np.nanmean(profiles_null, axis=0)
    s_epi, a_epi = _strength(epi)
    s_null, a_null = _strength(null)
    row = {
        "n_epicentres": int(np.mean([len(c) for c in centres_all])),
        "median_umi": float(np.median(library[valid])),
        "strength": s_epi - s_null,
        "amplitude": a_epi - a_null,
        **{f"epi_{lab}": v for lab, v in zip(RING_LABELS, epi)},
        **{f"null_{lab}": v for lab, v in zip(RING_LABELS, null)},
    }
    centres_union = np.unique(np.concatenate(centres_all)).astype(int)
    null_union = np.unique(np.concatenate([d for d in null_all if len(d)] or [np.array([], dtype=int)])).astype(int)
    core = np.zeros(len(points), dtype=bool)
    control = np.zeros(len(points), dtype=bool)
    if len(centres_union):
        core = valid & (cKDTree(points[centres_union]).query(points)[0] <= 150.0)
    if len(null_union):
        control = valid & ~core & (cKDTree(points[null_union]).query(points)[0] <= 150.0)
    return {"row": row, "core": core, "control": control, "smooth": smooth_halves, "centres": centres_all}


def run_organ(organ: str, samples, assignments, weights_table, composition: bool = False) -> dict:
    """Source scan and blind epicentre search for one organ."""
    rng = np.random.default_rng(1000 + SECTION_ORGANS.index(organ))
    group, genes, index, weights, halves, clock, random_sets = _sections(organ, samples, assignments, weights_table, rng)
    modules = {**SOURCES, **random_sets}
    covariates = {name: tuple(g for g in markers if g.lower() not in clock) for name, markers in COMPOSITION.items()}
    source_rows, blind_rows, core_rows, maps = [], [], [], {}
    print(f"  {organ}: {len(group)} sections, {len(weights)} weights", flush=True)
    for row in group.itertuples(index=False):
        loaded = load_section(Path(row.h5_path), Path(row.positions_path))
        if not np.array_equal(loaded["genes"], genes):
            raise ValueError(f"Gene order differs in {row.h5_path}")
        counts = loaded["counts"].tocsr()
        library = np.asarray(loaded["counts"].sum(axis=0)).ravel()
        points = np.column_stack([loaded["pxl_col"], loaded["pxl_row"]]) * _um_per_pixel(Path(row.h5_path))
        spots = assignments.loc[(assignments["organ"] == organ) & (assignments["mouse_id"] == int(row.mouse_id))]
        niche = spots.set_index("barcode").reindex(loaded["barcodes"])["niche"].to_numpy(dtype=float)
        valid = np.isfinite(niche) & (library > 0)
        levels = np.unique(niche[valid])
        columns = [(niche == level).astype(float) for level in levels] + [np.log(np.maximum(library, 1.0))]
        if composition:
            columns += [_module(counts, index, library, markers) for markers in covariates.values()]
        design = np.column_stack(columns)
        full = _score(counts, index, library, weights)
        residual = _residual(full, design, valid)
        meta = {"organ": organ, "mouse_id": int(row.mouse_id), "age_weeks": int(row.age_weeks), "age_months": float(row.age_months)}

        shown = {}
        for name, marker_genes in modules.items():
            module = _module(counts, index, library, marker_genes)
            if module is None:
                continue
            if name in MAP_MODULES:
                shown[name] = module
            kind = "random" if name.startswith("random_") else "candidate"
            source_rows.append({**meta, "module": name, "kind": kind, **_source_rho(module, residual, full, points, valid)})

        maps[(organ, int(row.mouse_id))] = {**meta, "x": loaded["pxl_col"], "y": loaded["pxl_row"], "residual": residual, **shown}
        blind = _blind_section(counts, index, library, halves, design, valid, niche, points, rng)
        blind_rows.append({**meta, **blind["row"]})
        core_rows.append(
            {
                **meta,
                "core": np.asarray(loaded["counts"] @ blind["core"].astype(float)).ravel(),
                "control": np.asarray(loaded["counts"] @ blind["control"].astype(float)).ravel(),
            }
        )
        maps[(organ, int(row.mouse_id))].update(
            {"smooth_a": blind["smooth"][0], "smooth_b": blind["smooth"][1], "centres_a": blind["centres"][0], "centres_b": blind["centres"][1]}
        )
    return {"source": source_rows, "blind": blind_rows, "core": core_rows, "maps": maps, "genes": genes, "clock": clock}


def _source_summary(sections: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for (organ, module), sub in sections.groupby(["organ", "module"]):
        kind = sub["kind"].iloc[0]
        x = sub["rho_adj"].to_numpy(float)
        age = sub["age_months"].to_numpy(float)
        row = {
            "organ": organ,
            "module": module,
            "kind": kind,
            "n": int(np.isfinite(x).sum()),
            "mean_rho_adj": float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
            "mean_rho_raw": float(np.nanmean(sub["rho_raw"])) if np.isfinite(sub["rho_raw"]).any() else np.nan,
        }
        for age_weeks in (17, 73, 112):
            row[f"rho_adj_{age_weeks}"] = float(np.nanmean(sub.loc[sub["age_weeks"] == age_weeks, "rho_adj"])) if (sub["age_weeks"] == age_weeks).any() else np.nan
        if kind == "candidate" and row["n"] >= 6:
            row["age_rho"], row["p_age"] = _perm_spearman(x, age, rng)
            row["p_mean"] = _sign_flip(x)
        else:
            row["age_rho"] = _spearman(x, age) if row["n"] >= 6 else np.nan
            row["p_age"] = np.nan
            row["p_mean"] = np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    out = []
    for organ, sub in table.groupby("organ"):
        random = sub.loc[sub["kind"] == "random"]
        sub = sub.copy()
        for metric in ("age_rho", "mean_rho_adj"):
            ref = random[metric].dropna().to_numpy()
            sub[f"rank_{metric}"] = [
                (np.sum(ref <= value) + 1) / (len(ref) + 1) if np.isfinite(value) and len(ref) else np.nan for value in sub[metric]
            ]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def _pooled_brain(sections: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    brain = sections.loc[sections["organ"].isin(BRAIN)]
    pooled = brain.groupby(keys + ["mouse_id", "age_weeks", "age_months"], as_index=False)[value].mean()
    pooled["organ"] = "Brain_pooled"
    return pooled


def _blind_summary(sections: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for organ, sub in sections.groupby("organ"):
        row = {"organ": organ, "n": len(sub)}
        for metric in ("strength", "amplitude"):
            x = sub[metric].to_numpy(float)
            row[f"mean_{metric}"] = float(np.nanmean(x))
            row[f"p_{metric}_mean"] = _sign_flip(x)
            row[f"age_rho_{metric}"], row[f"p_{metric}_age"] = _perm_spearman(x, sub["age_months"].to_numpy(float), rng)
            for age_weeks in (17, 73, 112):
                row[f"{metric}_{age_weeks}"] = float(np.nanmean(sub.loc[sub["age_weeks"] == age_weeks, metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def _core_genes(core_rows: list[dict], genes: np.ndarray, clock: set[str], label: str) -> pd.DataFrame:
    frame = pd.DataFrame(core_rows)
    if frame.empty:
        return pd.DataFrame()
    per_mouse = {}
    for mouse, sub in frame.groupby("mouse_id"):
        core = np.sum(np.stack(sub["core"].to_numpy()), axis=0)
        control = np.sum(np.stack(sub["control"].to_numpy()), axis=0)
        if core.sum() == 0 or control.sum() == 0:
            continue
        per_mouse[mouse] = (core / core.sum() * 1e6, control / control.sum() * 1e6, int(sub["age_weeks"].iloc[0]))
    if len(per_mouse) < 4:
        return pd.DataFrame()
    core_cpm = np.stack([v[0] for v in per_mouse.values()])
    ctrl_cpm = np.stack([v[1] for v in per_mouse.values()])
    ages = np.array([v[2] for v in per_mouse.values()], dtype=float)
    lfc = np.log2(core_cpm + 1.0) - np.log2(ctrl_cpm + 1.0)
    keep = (ctrl_cpm.mean(axis=0) >= 20.0) & np.array([g.lower() not in clock for g in genes])
    mean = lfc.mean(axis=0)
    sd = lfc.std(axis=0, ddof=1)
    t = mean / (sd / np.sqrt(lfc.shape[0]))
    age_rho = np.array([_spearman(lfc[:, j], ages) if keep[j] else np.nan for j in range(len(genes))])
    table = pd.DataFrame(
        {
            "set": label,
            "gene": genes,
            "n_mice": lfc.shape[0],
            "mean_log2fc": mean,
            "t": t,
            "frac_up": (lfc > 0).mean(axis=0),
            "control_cpm": ctrl_cpm.mean(axis=0),
            "age_rho_log2fc": age_rho,
        }
    )
    return table.loc[keep].sort_values("t", ascending=False).reset_index(drop=True)


def _write_source(rows: list[dict], suffix: str, rng) -> pd.DataFrame:
    source = pd.DataFrame(rows)
    source.to_csv(DEST / f"source_scan_sections{suffix}.csv", index=False)
    pooled = _pooled_brain(source.dropna(subset=["rho_adj"]), "rho_adj", ["module", "kind"])
    pooled["rho_raw"] = np.nan
    summary = _source_summary(pd.concat([source, pooled], ignore_index=True), rng)
    summary.to_csv(DEST / f"source_scan_summary{suffix}.csv", index=False)
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    DEST.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(OUT / "samples.csv")
    assignments = pd.read_csv(OUT / "niches" / "assignments.csv")
    weights_table = pd.read_csv(ROOT / "results" / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    rng = np.random.default_rng(29)
    passes = (("", SECTION_ORGANS, False), ("_composition", BRAIN, True))
    summaries = {}
    for suffix, organs, composition in passes:
        source_rows, blind_rows, maps, core_tables = [], [], {}, []
        brain_cores, brain_genes, brain_clock = [], None, set()
        for organ in organs:
            result = run_organ(organ, samples, assignments, weights_table, composition=composition)
            source_rows.extend(result["source"])
            blind_rows.extend(result["blind"])
            maps.update(result["maps"])
            core_tables.append(_core_genes(result["core"], result["genes"], result["clock"], organ))
            if organ in BRAIN:
                brain_cores.extend(result["core"])
                brain_genes = result["genes"]
                brain_clock |= result["clock"]
        source_summary = _write_source(source_rows, suffix, rng)
        np.savez_compressed(DEST / f"maps{suffix}.npz", maps=np.array(maps, dtype=object))

        blind = pd.DataFrame(blind_rows)
        blind.to_csv(DEST / f"blind_sections{suffix}.csv", index=False)
        pooled = blind.loc[blind["organ"].isin(BRAIN)].groupby(["mouse_id", "age_weeks", "age_months"], as_index=False)[["strength", "amplitude"]].mean()
        pooled["organ"] = "Brain_pooled"
        blind_summary = _blind_summary(pd.concat([blind, pooled], ignore_index=True), rng)
        blind_summary.to_csv(DEST / f"blind_summary{suffix}.csv", index=False)

        core_tables.append(_core_genes(brain_cores, brain_genes, brain_clock, "Brain_pooled"))
        pd.concat([t for t in core_tables if not t.empty], ignore_index=True).to_csv(DEST / f"epicentre_core_genes{suffix}.csv", index=False)
        summaries[suffix] = (source_summary, blind_summary)

    pd.set_option("display.width", 220)
    shown = ["organ", "module", "n", "mean_rho_adj", "rank_mean_rho_adj", "p_mean", "age_rho", "p_age", "rank_age_rho"]
    for suffix, label in (("", "niche and depth"), ("_composition", "niche, depth and composition")):
        source_summary, blind_summary = summaries[suffix]
        print(f"\nSource scan, candidates, adjusted for {label}")
        print(source_summary.loc[source_summary["kind"] == "candidate", shown].round(3).to_string(index=False))
        print(f"\nBlind epicentres, adjusted for {label}")
        print(blind_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
