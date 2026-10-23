"""Exploratory spatial questions that do not retest the published clock claims.

The mouse is the unit throughout. Niche timing uses the saved niche scores.
Anatomy, patchiness and immune-neighbourhood tests rescore every section.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from .spatial_frozen import _weights_for
from .spatial_io import SECTION_ORGANS, load_section

from .paths import project_root

ROOT = project_root()
OUT = ROOT / "results" / "spatial_aging"
DEST = OUT / "followup"
TECHNICAL = {"Gm42418", "Lars2", "Malat1"}
N_PERM = 5_000
RNG = np.random.default_rng(17)
SPOT_UM = 55.0

PORTAL = ("Cyp2f2", "Sds", "Hsd17b13")
CENTRAL = ("Glul", "Cyp2e1", "Oat")
CORTEX_MARKERS = ("Slc34a1", "Lrp2")
MEDULLA_MARKERS = ("Umod", "Slc12a1", "Aqp2")
WM_MARKERS = ("Plp1", "Mbp", "Mobp")
NEURON_MARKERS = ("Snap25", "Nrgn", "Camk2a")
IMMUNE = ("Ptprc", "Cd3e", "Cd79a", "C1qa", "Lcn2")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


def _perm_p(x: np.ndarray, y: np.ndarray, observed: float) -> float:
    if not np.isfinite(observed):
        return float("nan")
    count = 0
    finite = 0
    for _ in range(N_PERM):
        rho = _spearman(x, RNG.permutation(y))
        if np.isfinite(rho):
            finite += 1
            if abs(rho) >= abs(observed):
                count += 1
    return (count + 1) / (finite + 1) if finite else float("nan")


def _bh(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index)
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    n = len(ranked)
    adjusted = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running
    out = pd.Series(np.nan, index=p_values.index)
    out.loc[valid.index[order]] = adjusted
    return out


def _technical(markers: str) -> bool:
    genes = {gene.strip() for gene in str(markers).split(",")}
    return bool(genes & TECHNICAL)


def _score_matrix(counts: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    """Frozen score: log2(CPM+1), gene-wise z across columns, then the weight sum."""
    library = counts.sum(axis=0).replace(0, np.nan)
    logged = np.log2(counts.divide(library, axis=1) * 1e6 + 1.0)
    block = logged.reindex(weights.index)
    values = block.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    std[std == 0] = np.nan
    z = (values - mean) / std
    use = np.isfinite(z).all(axis=1) & np.isfinite(weights.to_numpy())
    if int(use.sum()) < 3:
        return np.full(counts.shape[1], np.nan)
    coef = weights.to_numpy()[use]
    return z[use].T @ coef


def niche_timing() -> pd.DataFrame:
    scores = pd.read_csv(OUT / "niches" / "scores.csv")
    metrics = pd.read_csv(OUT / "niches" / "metrics.csv")
    rows = []
    for (organ, niche), sub in scores.groupby(["organ", "niche"]):
        by_age = {int(age): frame["score"].to_numpy(dtype=float) for age, frame in sub.groupby("age_weeks")}
        if any(age not in by_age or len(by_age[age]) < 2 for age in (17, 73, 112)):
            continue
        early = float(by_age[73].mean() - by_age[17].mean())
        late = float(by_age[112].mean() - by_age[73].mean())
        total = float(by_age[112].mean() - by_age[17].mean())
        timing = early / total if abs(total) > 1e-8 else float("nan")
        labels = np.concatenate([np.full(len(by_age[age]), age) for age in (17, 73, 112)])
        values = np.concatenate([by_age[age] for age in (17, 73, 112)])
        null_early = np.empty(N_PERM)
        null_late = np.empty(N_PERM)
        for i in range(N_PERM):
            shuffled = RNG.permutation(labels)
            means = {age: values[shuffled == age].mean() for age in (17, 73, 112)}
            null_early[i] = means[73] - means[17]
            null_late[i] = means[112] - means[73]
        boot = np.empty(N_PERM)
        for i in range(N_PERM):
            means = {}
            for age in (17, 73, 112):
                draw = RNG.choice(by_age[age], size=len(by_age[age]), replace=True)
                means[age] = draw.mean()
            change = means[112] - means[17]
            boot[i] = (means[73] - means[17]) / change if abs(change) > 1e-8 else np.nan
        meta = metrics.loc[(metrics["organ"] == organ) & (metrics["niche"] == niche)].iloc[0]
        rows.append(
            {
                "organ": organ,
                "niche": int(niche),
                "markers": meta["markers"],
                "technical": _technical(meta["markers"]),
                "n_17": int(len(by_age[17])),
                "n_73": int(len(by_age[73])),
                "n_112": int(len(by_age[112])),
                "early": early,
                "late": late,
                "total": total,
                "timing": timing,
                "timing_low": float(np.nanpercentile(boot, 2.5)),
                "timing_high": float(np.nanpercentile(boot, 97.5)),
                "p_early": (np.sum(np.abs(null_early) >= abs(early)) + 1) / (N_PERM + 1),
                "p_late": (np.sum(np.abs(null_late) >= abs(late)) + 1) / (N_PERM + 1),
                "score_rho": float(meta["spearman_score_age"]),
                "fraction_rho": float(meta["spearman_fraction_age"]),
            }
        )
    table = pd.DataFrame(rows)
    biological = ~table["technical"]
    table.loc[biological, "q_early"] = _bh(table.loc[biological, "p_early"])
    table.loc[biological, "q_late"] = _bh(table.loc[biological, "p_late"])
    table.to_csv(DEST / "niche_timing.csv", index=False)
    return table


def score_spread() -> pd.DataFrame:
    summary = pd.read_csv(OUT / "visium_spot_summary.csv")
    summary["spread"] = summary["p90_spot_score"] - summary["p10_spot_score"]
    rows = []
    for organ, sub in summary.groupby("organ"):
        rho = _spearman(sub["spread"].to_numpy(float), sub["age_months"].to_numpy(float))
        rows.append(
            {
                "organ": organ,
                "n": int(len(sub)),
                "spearman_spread_age": rho,
                "p": _perm_p(sub["spread"].to_numpy(float), sub["age_months"].to_numpy(float), rho),
                "mean_spread_17": float(sub.loc[sub["age_weeks"] == 17, "spread"].mean()),
                "mean_spread_112": float(sub.loc[sub["age_weeks"] == 112, "spread"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "score_spread.csv", index=False)
    return table


def _wanted_genes(organ: str, weights: pd.Series) -> list[str]:
    extra = list(IMMUNE)
    if organ == "Liver":
        extra += list(PORTAL) + list(CENTRAL)
    elif organ == "Kidney":
        extra += list(CORTEX_MARKERS) + list(MEDULLA_MARKERS)
    elif str(organ).startswith("Brain"):
        extra += list(WM_MARKERS) + list(NEURON_MARKERS)
    ordered = list(dict.fromkeys([*weights.index.astype(str), *extra]))
    return ordered


def _extract(loaded: dict, wanted: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    lookup = {gene.lower(): i for i, gene in enumerate(loaded["genes"])}
    rows = []
    names = []
    for gene in wanted:
        hit = lookup.get(gene.lower())
        if hit is not None:
            rows.append(hit)
            names.append(gene)
    counts = loaded["counts"][rows].toarray()
    library = np.asarray(loaded["counts"].sum(axis=0)).ravel()
    return names, counts, library


def _module(names: list[str], log_cp10k: np.ndarray, markers: tuple[str, ...]) -> np.ndarray:
    index = {gene.lower(): i for i, gene in enumerate(names)}
    rows = [index[gene.lower()] for gene in markers if gene.lower() in index]
    if not rows:
        return np.full(log_cp10k.shape[1], np.nan)
    return log_cp10k[rows].mean(axis=0)


def _hex_neighbors(rows: np.ndarray, cols: np.ndarray) -> list[tuple[int, int]]:
    location = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    pairs = []
    for i, (r, c) in enumerate(zip(rows.astype(int), cols.astype(int))):
        if r % 2 == 0:
            candidates = ((r, c - 1), (r, c + 1), (r - 1, c), (r - 1, c - 1), (r + 1, c), (r + 1, c - 1))
        else:
            candidates = ((r, c - 1), (r, c + 1), (r - 1, c), (r - 1, c + 1), (r + 1, c), (r + 1, c + 1))
        for key in candidates:
            j = location.get(key)
            if j is not None and i < j:
                pairs.append((i, j))
    return pairs


def _moran(values: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if len(pairs) < 20:
        return float("nan")
    x = values - np.nanmean(values)
    if not np.isfinite(x).all() or np.sum(x**2) <= 0:
        return float("nan")
    cross = 0.0
    for i, j in pairs:
        cross += x[i] * x[j]
    n = len(values)
    w = len(pairs)
    return float(n / w * cross / np.sum(x**2))


def _within_score(log_cpm: np.ndarray, names: list[str], weights: pd.Series) -> np.ndarray:
    index = {gene.lower(): i for i, gene in enumerate(names)}
    rows = []
    coef = []
    for gene, weight in weights.items():
        hit = index.get(str(gene).lower())
        if hit is not None and np.isfinite(weight):
            rows.append(hit)
            coef.append(float(weight))
    if len(rows) < 3:
        return np.full(log_cpm.shape[1], np.nan)
    block = log_cpm[rows]
    mean = block.mean(axis=1, keepdims=True)
    std = block.std(axis=1, keepdims=True)
    std[std == 0] = np.nan
    z = (block - mean) / std
    coef = np.asarray(coef)
    use = np.isfinite(z).all(axis=1)
    return z[use].T @ coef[use]


def _um_per_pixel(h5_path: Path) -> float:
    path = h5_path.with_name(h5_path.name.replace("filtered_feature_bc_matrix.h5", "scalefactors_json.json.gz"))
    with gzip.open(path, "rt") as handle:
        factors = json.load(handle)
    return SPOT_UM / float(factors["spot_diameter_fullres"])


def spatial_pass(weights_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(OUT / "samples.csv")
    moran_rows = []
    immune_rows = []
    compartment_counts: dict[tuple, list] = {}
    structure_counts: dict[tuple, list] = {}
    for organ in SECTION_ORGANS:
        group = samples.loc[samples["organ"] == organ]
        probe = load_section(Path(group.iloc[0].h5_path), Path(group.iloc[0].positions_path))
        lookup = {gene.lower(): gene for gene in probe["genes"]}
        weights, _n = _weights_for(weights_table, organ, lookup)
        wanted = _wanted_genes(organ, weights)
        print(f"  {organ}: {len(group)} sections, {len(weights)} weights", flush=True)
        for row in group.itertuples(index=False):
            loaded = load_section(Path(row.h5_path), Path(row.positions_path))
            names, raw, library = _extract(loaded, wanted)
            scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
            log_cp10k = np.log1p(raw * scale)
            cpm_scale = np.divide(1e6, library, out=np.zeros_like(library), where=library > 0)
            log_cpm = np.log2(raw * cpm_scale + 1.0)
            positions = pd.read_csv(row.positions_path)
            coordinates = positions.set_index("barcode").reindex(loaded["barcodes"])
            array_row = coordinates["array_row"].to_numpy(dtype=float)
            array_col = coordinates["array_col"].to_numpy(dtype=float)
            score = _within_score(log_cpm, names, weights)
            pairs = _hex_neighbors(array_row, array_col)
            moran_rows.append(
                {
                    "organ": organ,
                    "mouse_id": int(row.mouse_id),
                    "age_weeks": int(row.age_weeks),
                    "age_months": float(row.age_months),
                    "moran": _moran(score, pairs),
                    "n_spots": int(len(score)),
                }
            )
            _collect_axis(organ, row, names, raw, log_cp10k, compartment_counts, structure_counts)
            immune_rows.append(_immune_row(organ, row, names, log_cp10k, score, loaded, Path(row.h5_path)))
    compartments = _score_compartments(compartment_counts, weights_table)
    structures = _score_structures(structure_counts, weights_table)
    moran = pd.DataFrame(moran_rows)
    moran_summary = _associate(moran, "moran", "moran")
    immune = pd.DataFrame([row for row in immune_rows if row])
    immune_summary = _associate(immune, "distance_rho", "immune_distance") if not immune.empty else pd.DataFrame()
    moran.to_csv(DEST / "moran_sections.csv", index=False)
    moran_summary.to_csv(DEST / "moran_vs_age.csv", index=False)
    immune.to_csv(DEST / "immune_distance_sections.csv", index=False)
    immune_summary.to_csv(DEST / "immune_distance_vs_age.csv", index=False)
    return moran_summary, immune_summary, compartments, structures


def _collect_axis(organ, row, names, raw, log_cp10k, compartments, structures) -> None:
    if organ == "Liver":
        axis = _module(names, log_cp10k, PORTAL) - _module(names, log_cp10k, CENTRAL)
        _store_ends(compartments, ("Liver", "portal"), ("Liver", "central"), row, names, raw, axis)
    elif organ == "Kidney":
        axis = _module(names, log_cp10k, CORTEX_MARKERS) - _module(names, log_cp10k, MEDULLA_MARKERS)
        _store_ends(compartments, ("Kidney", "cortex"), ("Kidney", "medulla"), row, names, raw, axis)
    elif str(organ).startswith("Brain"):
        wm = _module(names, log_cp10k, WM_MARKERS)
        neuron = _module(names, log_cp10k, NEURON_MARKERS)
        if np.isfinite(wm).sum() < 50:
            return
        wm_cut = np.nanquantile(wm, 0.80)
        neuron_cut = np.nanquantile(neuron, 0.80) if np.isfinite(neuron).any() else np.inf
        wm_mask = wm >= wm_cut
        neuron_mask = (neuron >= neuron_cut) & ~wm_mask
        _store_mask(structures, (organ, "white_matter"), row, names, raw, wm_mask)
        _store_mask(structures, (organ, "neuronal"), row, names, raw, neuron_mask)


def _store_ends(store, high_key, low_key, row, names, raw, axis) -> None:
    finite = np.isfinite(axis)
    if finite.sum() < 100:
        return
    low, high = np.nanquantile(axis[finite], [0.25, 0.75])
    _store_mask(store, high_key, row, names, raw, finite & (axis >= high))
    _store_mask(store, low_key, row, names, raw, finite & (axis <= low))


def _store_mask(store, key, row, names, raw, mask) -> None:
    if int(mask.sum()) < 30:
        return
    store.setdefault(key, []).append(
        {
            "mouse_id": int(row.mouse_id),
            "age_weeks": int(row.age_weeks),
            "age_months": float(row.age_months),
            "names": names,
            "counts": raw[:, mask].sum(axis=1),
            "n_spots": int(mask.sum()),
        }
    )


def _score_saved(records: list[dict], weights: pd.Series) -> pd.DataFrame:
    kept = []
    seen = set()
    for record in records:
        if record["mouse_id"] in seen:
            continue
        seen.add(record["mouse_id"])
        kept.append(record)
    names = kept[0]["names"]
    counts = pd.DataFrame(
        np.column_stack([record["counts"] for record in kept]),
        index=names,
        columns=[str(record["mouse_id"]) for record in kept],
    )
    scored = _score_matrix(counts, weights)
    return pd.DataFrame(
        {
            "mouse_id": [int(record["mouse_id"]) for record in kept],
            "age_weeks": [int(record["age_weeks"]) for record in kept],
            "age_months": [float(record["age_months"]) for record in kept],
            "n_spots": [int(record["n_spots"]) for record in kept],
            "score": scored,
        }
    )


def _score_compartments(store, weights_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    per_mouse = []
    for (organ, end), records in store.items():
        probe_genes = {gene.lower(): gene for gene in records[0]["names"]}
        weights, _n = _weights_for(weights_table, organ, probe_genes)
        scored = _score_saved(records, weights)
        scored["organ"] = organ
        scored["end"] = end
        per_mouse.append(scored)
        rho = _spearman(scored["score"].to_numpy(float), scored["age_months"].to_numpy(float))
        rows.append(
            {
                "organ": organ,
                "end": end,
                "n": int(len(scored)),
                "mean_spots": float(scored["n_spots"].mean()),
                "spearman": rho,
                "p": _perm_p(scored["score"].to_numpy(float), scored["age_months"].to_numpy(float), rho),
            }
        )
    mouse = pd.concat(per_mouse, ignore_index=True) if per_mouse else pd.DataFrame()
    if not mouse.empty:
        for organ in mouse["organ"].unique():
            wide = mouse.loc[mouse["organ"] == organ].pivot(index="mouse_id", columns="end", values="score")
            ages = mouse.loc[mouse["organ"] == organ].drop_duplicates("mouse_id").set_index("mouse_id")["age_months"]
            ends = list(wide.columns)
            if len(ends) == 2:
                delta = (wide[ends[0]] - wide[ends[1]]).dropna()
                age = ages.reindex(delta.index).to_numpy(float)
                rho = _spearman(delta.to_numpy(float), age)
                rows.append(
                    {
                        "organ": organ,
                        "end": f"{ends[0]}_minus_{ends[1]}",
                        "n": int(len(delta)),
                        "mean_spots": np.nan,
                        "spearman": rho,
                        "p": _perm_p(delta.to_numpy(float), age, rho),
                    }
                )
        mouse.to_csv(DEST / "anatomy_scores.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "anatomy_vs_age.csv", index=False)
    return table


def _score_structures(store, weights_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frames = []
    for (organ, kind), records in store.items():
        probe_genes = {gene.lower(): gene for gene in records[0]["names"]}
        weights, _n = _weights_for(weights_table, organ, probe_genes)
        scored = _score_saved(records, weights)
        scored["organ"] = organ
        scored["structure"] = kind
        frames.append(scored)
        rho = _spearman(scored["score"].to_numpy(float), scored["age_months"].to_numpy(float))
        rows.append(
            {
                "organ": organ,
                "structure": kind,
                "n": int(len(scored)),
                "mean_spots": float(scored["n_spots"].mean()),
                "spearman": rho,
                "p": _perm_p(scored["score"].to_numpy(float), scored["age_months"].to_numpy(float), rho),
            }
        )
    table = pd.DataFrame(rows)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(DEST / "brain_structure_scores.csv", index=False)
    table.to_csv(DEST / "brain_structure_vs_age.csv", index=False)
    return table


def _nearest(sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # Chunked so a section does not build a full target-by-source matrix.
    nearest = np.empty(targets.shape[0])
    for start in range(0, len(targets), 400):
        block = targets[start : start + 400]
        distance = np.sqrt(((block[:, None, :] - sources[None, :, :]) ** 2).sum(axis=2))
        nearest[start : start + block.shape[0]] = distance.min(axis=1)
    return nearest


def _immune_row(organ, row, names, log_cp10k, score, loaded, h5_path: Path) -> dict:
    module = _module(names, log_cp10k, IMMUNE)
    finite = np.isfinite(module) & np.isfinite(score)
    if finite.sum() < 200:
        return {}
    cutoff = np.nanquantile(module[finite], 0.90)
    immune = finite & (module >= cutoff) & (module > 0)
    if immune.sum() < 30 or immune.mean() > 0.4:
        return {}
    um = _um_per_pixel(h5_path)
    points = np.column_stack([loaded["pxl_col"], loaded["pxl_row"]]) * um
    other = ~immune & finite
    if other.sum() < 100:
        return {}
    nearest = _nearest(points[immune], points[other])
    near = nearest <= 750
    if near.sum() < 80:
        return {}
    rho = _spearman(score[other][near], nearest[near])
    return {
        "organ": organ,
        "mouse_id": int(row.mouse_id),
        "age_weeks": int(row.age_weeks),
        "age_months": float(row.age_months),
        "n_immune": int(immune.sum()),
        "distance_rho": rho,
    }


def _associate(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    for organ, sub in frame.groupby("organ"):
        x = sub[column].to_numpy(float)
        y = sub["age_months"].to_numpy(float)
        rho = _spearman(x, y)
        rows.append({"test": label, "organ": organ, "n": int(np.isfinite(x).sum()), "spearman": rho, "p": _perm_p(x, y, rho)})
    return pd.DataFrame(rows)


def _timing_class(row: pd.Series) -> str:
    if row["technical"] or not np.isfinite(row["timing"]):
        return "excluded"
    if row["total"] <= 0:
        return "not_increasing"
    if row["timing"] >= 0.67 and row["p_early"] < 0.05 and row["p_late"] >= 0.05:
        return "early"
    if row["timing"] <= 0.33 and row["p_late"] < 0.05 and row["p_early"] >= 0.05:
        return "late"
    if row["p_early"] < 0.05 and row["p_late"] < 0.05:
        return "both"
    return "undetermined"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("Niche timing", flush=True)
    timing = niche_timing()
    timing["class"] = timing.apply(_timing_class, axis=1)
    timing.to_csv(DEST / "niche_timing.csv", index=False)
    print(timing.loc[~timing["technical"], "class"].value_counts().to_string(), flush=True)
    print("Score spread", flush=True)
    spread = score_spread()
    print(spread.to_string(index=False), flush=True)
    print("Spatial pass", flush=True)
    weights = pd.read_csv(ROOT / "results" / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    moran, immune, anatomy, structures = spatial_pass(weights)
    print("Moran", flush=True)
    print(moran.to_string(index=False), flush=True)
    print("Immune", flush=True)
    print(immune.to_string(index=False) if len(immune) else "none", flush=True)
    print("Anatomy", flush=True)
    print(anatomy.to_string(index=False), flush=True)
    print("Brain structures", flush=True)
    print(structures.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
