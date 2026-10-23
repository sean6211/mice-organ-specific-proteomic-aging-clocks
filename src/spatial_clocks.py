"""Mouse-level transcriptomic age models for GSE283283 Visium sections.

Each section is one mouse. Models are elastic nets tested by holding out one mouse.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest, fisher_exact, spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .ageaccel import ageaccel_l
from .spatial_io import SECTION_ORGANS
from .train_clocks import ALPHAS, L1_RATIOS, RANDOM_STATE, _metrics

N_PERM = 200
STABLE_CUTOFF = 0.5
TOP_GENES = 5_000
MIN_CPM = 1.0
MIN_DETECT = 0.8
BRAIN_LEVELS = ["Brain1", "Brain2", "Brain3", "Brain4", "Brain5"]
SCIFO_ORGAN = {
    "Brain": "brain",
    "Brain1": "brain",
    "Brain2": "brain",
    "Brain3": "brain",
    "Brain4": "brain",
    "Brain5": "brain",
    "Heart": "heart",
    "Kidney": "kidney",
    "Liver": "liver",
    "Lung": "lung",
    "Spleen": "spleen",
}


def log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    values = counts.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    library = values.sum(axis=0).replace(0.0, np.nan)
    return np.log2(values.div(library, axis=1) * 1e6 + 1.0)


def detection_fraction(counts: pd.DataFrame) -> pd.Series:
    values = counts.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    library = values.sum(axis=0).replace(0.0, np.nan)
    cpm = values.div(library, axis=1) * 1e6
    return (cpm >= MIN_CPM).mean(axis=1)


def select_genes(log_values: pd.DataFrame, eligible: pd.Series) -> list[str]:
    """CPM and variance filter. Neither step uses age."""
    fraction = eligible.reindex(log_values.index)
    keep = fraction.fillna(0).astype(float) >= MIN_DETECT
    varying = log_values.loc[keep]
    variance = varying.var(axis=1, ddof=0).sort_values(ascending=False)
    variance = variance[variance > 1e-12]
    return variance.index[:TOP_GENES].astype(str).tolist()


def _pipeline(n_jobs: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=list(L1_RATIOS),
                    alphas=ALPHAS,
                    cv=5,
                    max_iter=8000,
                    tol=1e-3,
                    n_jobs=n_jobs,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


def _lomo_predict(x: np.ndarray, y: np.ndarray, n_jobs: int, collect: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    preds = np.full(len(y), np.nan)
    coefs = []
    for train_idx, test_idx in LeaveOneOut().split(x):
        model = _pipeline(n_jobs)
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])
        if collect:
            coefs.append(model.named_steps["model"].coef_.copy())
    stacked = np.vstack(coefs) if collect else None
    return preds, stacked


def _perm_rho(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    y_perm = np.random.default_rng(seed).permutation(y)
    preds, _ = _lomo_predict(x, y_perm, n_jobs=1, collect=False)
    return _spearman(preds, y_perm)


def pooled_brain(level_counts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    """Average brain levels after subtracting each level's gene mean.

    A gene is eligible when CPM is at least 1 in 80% of all brain sections.
    """
    mice = sorted(set().union(*[set(frame.columns.astype(str)) for frame in level_counts.values()]), key=int)
    genes = level_counts[BRAIN_LEVELS[0]].index.astype(str)
    centered = []
    fractions = []
    weights = []
    for organ in BRAIN_LEVELS:
        counts = level_counts[organ].copy()
        counts.columns = counts.columns.astype(str)
        fractions.append(detection_fraction(counts).reindex(genes).fillna(0.0))
        weights.append(counts.shape[1])
        logged = log_cpm(counts).reindex(index=genes, columns=mice)
        centered.append(logged.sub(logged.mean(axis=1), axis=0))
    stacked = np.stack([frame.to_numpy(dtype=float) for frame in centered], axis=0)
    with np.errstate(all="ignore"):
        pooled = np.nanmean(stacked, axis=0)
    profile = pd.DataFrame(pooled, index=genes, columns=mice)
    profile = profile.dropna(axis=1, how="all")
    fraction = sum(frame * weight for frame, weight in zip(fractions, weights)) / sum(weights)
    return profile, fraction


def _xy(log_values: pd.DataFrame, genes: list[str], samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = samples.sort_values(["age_months", "mouse_id"])
    columns = [str(mouse) for mouse in order["mouse_id"]]
    block = log_values.reindex(index=genes, columns=columns).T
    return (
        block.to_numpy(dtype=float),
        order["age_months"].to_numpy(dtype=float),
        order["mouse_id"].to_numpy(dtype=int),
    )


def _stability(genes: list[str], coefs: np.ndarray, organ: str) -> pd.DataFrame:
    selected = np.abs(coefs) > 1e-12
    present = selected.any(axis=0)
    mean = np.zeros(coefs.shape[1], dtype=float)
    mean[present] = coefs.sum(axis=0)[present] / np.maximum(selected.sum(axis=0)[present], 1)
    frame = pd.DataFrame(
        {
            "organ": organ,
            "gene": genes,
            "selection_frequency": selected.mean(axis=0),
            "mean_coef": mean,
            "n_folds": coefs.shape[0],
        }
    )
    return frame.sort_values(["selection_frequency", "gene"], ascending=[False, True]).reset_index(drop=True)


def _middle_age(x: np.ndarray, y: np.ndarray, weeks: np.ndarray, organ: str) -> dict:
    train = np.isin(weeks, [17, 112])
    test = weeks == 73
    model = _pipeline(n_jobs=-1)
    model.fit(x[train], y[train])
    pred = np.full(len(y), np.nan)
    pred[test] = model.predict(x[test])
    pred[train] = model.predict(x[train])
    return {
        "organ": organ,
        "n_train": int(train.sum()),
        "n_middle": int(test.sum()),
        "mean_age_middle": float(np.mean(y[test])) if test.any() else np.nan,
        "mean_pred_middle": float(np.mean(pred[test])) if test.any() else np.nan,
        "mae_middle": float(np.mean(np.abs(pred[test] - y[test]))) if test.any() else np.nan,
        "mean_pred_young": float(np.mean(pred[weeks == 17])),
        "mean_pred_old": float(np.mean(pred[weeks == 112])),
    }


def _overlap(stability: pd.DataFrame, scifo: pd.DataFrame, organ: str, tested: list[str]) -> dict:
    scifo_organ = SCIFO_ORGAN[organ]
    reference = scifo.loc[scifo["organ"] == scifo_organ].copy()
    reference["gene_key"] = reference["gene"].astype(str).str.lower()
    reference = reference.drop_duplicates("gene_key")
    tested_keys = pd.Index([gene.lower() for gene in tested])
    stable = stability.loc[stability["selection_frequency"] > STABLE_CUTOFF].copy()
    stable["gene_key"] = stable["gene"].astype(str).str.lower()
    universe = set(tested_keys)
    scifo_in = set(reference["gene_key"]).intersection(universe)
    stable_in = set(stable["gene_key"]).intersection(universe)
    both = stable_in & scifo_in
    table = np.array(
        [
            [len(both), len(scifo_in - stable_in)],
            [len(stable_in - scifo_in), len(universe - scifo_in - stable_in)],
        ]
    )
    odds, p_fisher = fisher_exact(table, alternative="greater")
    coef = stable.drop_duplicates("gene_key").set_index("gene_key")["mean_coef"]
    ref_coef = reference.set_index("gene_key")["mean_coef"]
    shared = sorted(both)
    same = 0
    if shared:
        spatial_sign = np.sign(coef.reindex(shared).to_numpy(dtype=float))
        protein_sign = np.sign(ref_coef.reindex(shared).to_numpy(dtype=float))
        comparable = (spatial_sign != 0) & (protein_sign != 0)
        same = int(np.sum(spatial_sign[comparable] == protein_sign[comparable]))
        n_sign = int(comparable.sum())
    else:
        n_sign = 0
    sign_p = float(binomtest(same, n_sign, 0.5).pvalue) if n_sign else np.nan
    return {
        "organ": organ,
        "scifo_organ": scifo_organ,
        "n_tested": len(universe),
        "n_stable": len(stable_in),
        "n_scifo_in_tested": len(scifo_in),
        "n_overlap": len(both),
        "odds_ratio": float(odds),
        "fisher_p": float(p_fisher),
        "n_sign_compared": n_sign,
        "n_same_sign": same,
        "sign_binom_p": sign_p,
    }


def train_organ_clock(
    organ: str,
    log_values: pd.DataFrame,
    eligible: pd.Series,
    samples: pd.DataFrame,
    scifo: pd.DataFrame,
    out_dir: Path,
    n_perm: int = N_PERM,
) -> tuple[dict, pd.DataFrame, dict, dict]:
    genes = select_genes(log_values, eligible)
    organ_samples = samples.loc[samples["organ"] == organ].copy()
    if organ == "Brain":
        organ_samples = (
            samples.loc[samples["organ"].isin(BRAIN_LEVELS), ["mouse_id", "age_weeks", "age_months"]]
            .drop_duplicates("mouse_id")
            .copy()
        )
        organ_samples["organ"] = "Brain"
    x, y, mice = _xy(log_values, genes, organ_samples)
    weeks = organ_samples.sort_values(["age_months", "mouse_id"])["age_weeks"].to_numpy(dtype=int)
    print(f"  {organ}: {len(genes)} genes, {len(y)} mice", flush=True)
    preds, coefs = _lomo_predict(x, y, n_jobs=1, collect=True)
    if coefs is None:
        raise RuntimeError(organ)
    metrics = {"organ": organ, "n_mice": int(len(y)), "n_genes": int(len(genes)), **_metrics(y, preds)}
    metrics["spearman_r"] = _spearman(preds, y)
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    nulls = Parallel(n_jobs=n_workers)(delayed(_perm_rho)(x, y, 20_000 + i) for i in range(n_perm))
    nulls = np.asarray(nulls, dtype=float)
    n_ge = int(np.sum(nulls[np.isfinite(nulls)] >= metrics["spearman_r"]))
    metrics["spearman_p_perm"] = (n_ge + 1) / (int(np.isfinite(nulls).sum()) + 1)
    metrics["null_mean"] = float(np.nanmean(nulls))
    print(
        f"    LOMO Spearman={metrics['spearman_r']:.3f}  p={metrics['spearman_p_perm']:.4g}  "
        f"MAE={metrics['mae']:.2f} mo",
        flush=True,
    )
    stability = _stability(genes, coefs, organ)
    full = _pipeline(n_jobs=-1)
    full.fit(x, y)
    coefficients = pd.DataFrame(
        {"organ": organ, "gene": genes, "coefficient": full.named_steps["model"].coef_}
    )
    coefficients = coefficients.loc[coefficients["coefficient"].abs() > 1e-12].sort_values(
        "coefficient", key=lambda series: series.abs(), ascending=False
    )
    stability.to_csv(out_dir / "clocks" / "stability" / f"{organ}.csv", index=False)
    coefficients.to_csv(out_dir / "clocks" / "coefficients" / f"{organ}.csv", index=False)
    predictions = pd.DataFrame(
        {
            "organ": organ,
            "mouse_id": mice,
            "age_weeks": weeks,
            "age_months": y,
            "pred_loo": preds,
            "gap_loo": preds - y,
        }
    )
    middle = _middle_age(x, y, weeks, organ)
    overlap = _overlap(stability, scifo, organ, genes)
    null_frame = pd.DataFrame({"organ": organ, "null_spearman": nulls})
    return metrics, predictions, middle, overlap, null_frame


def middle_holdout_predictions(out_dir: Path) -> pd.DataFrame:
    """Predict every mouse from the model trained on 17- and 112-week mice only.

    Same gene filter and elastic net as the published middle-age summary.
    """
    samples = pd.read_csv(out_dir / "samples.csv")
    level_counts = {organ: load_organ_counts(out_dir, organ) for organ in BRAIN_LEVELS}
    brain_log, brain_fraction = pooled_brain(level_counts)
    jobs: list[tuple[str, pd.DataFrame, pd.Series]] = [("Brain", brain_log, brain_fraction)]
    for organ in SECTION_ORGANS:
        counts = level_counts[organ] if organ in level_counts else load_organ_counts(out_dir, organ)
        jobs.append((organ, log_cpm(counts), detection_fraction(counts)))
    frames = []
    for organ, logged, eligible in jobs:
        print(f"  middle holdout {organ}", flush=True)
        genes = select_genes(logged, eligible)
        if organ == "Brain":
            organ_samples = (
                samples.loc[samples["organ"].isin(BRAIN_LEVELS), ["mouse_id", "age_weeks", "age_months"]]
                .drop_duplicates("mouse_id")
                .copy()
            )
            organ_samples["organ"] = "Brain"
        else:
            organ_samples = samples.loc[samples["organ"] == organ].copy()
        x, y, mice = _xy(logged, genes, organ_samples)
        order = organ_samples.sort_values(["age_months", "mouse_id"])
        weeks = order["age_weeks"].to_numpy(dtype=int)
        train = np.isin(weeks, [17, 112])
        model = _pipeline(n_jobs=-1)
        model.fit(x[train], y[train])
        pred = np.full(len(y), np.nan)
        test = weeks == 73
        pred[test] = model.predict(x[test])
        pred[train] = model.predict(x[train])
        frames.append(
            pd.DataFrame(
                {
                    "organ": organ,
                    "mouse_id": mice,
                    "age_weeks": weeks,
                    "age_months": y,
                    "pred": pred,
                    "role": np.where(test, "heldout", "train"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_organ_counts(out_dir: Path, organ: str) -> pd.DataFrame:
    frame = pd.read_csv(out_dir / "pseudobulk" / f"{organ}.csv", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def write_clocks(out_dir: Path, results_root: Path, n_perm: int = N_PERM) -> pd.DataFrame:
    clock_dir = out_dir / "clocks"
    (clock_dir / "stability").mkdir(parents=True, exist_ok=True)
    (clock_dir / "coefficients").mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(out_dir / "samples.csv")
    scifo = pd.read_csv(results_root / "human_clock_overlap" / "mouse_stable_with_genes.csv")
    metric_rows = []
    pred_frames = []
    middle_rows = []
    overlap_rows = []
    null_frames = []
    level_counts = {organ: load_organ_counts(out_dir, organ) for organ in BRAIN_LEVELS}
    brain_log, brain_fraction = pooled_brain(level_counts)
    jobs: list[tuple[str, pd.DataFrame, pd.Series]] = [("Brain", brain_log, brain_fraction)]
    for organ in SECTION_ORGANS:
        counts = level_counts[organ] if organ in level_counts else load_organ_counts(out_dir, organ)
        jobs.append((organ, log_cpm(counts), detection_fraction(counts)))
    for organ, logged, eligible in jobs:
        metrics, predictions, middle, overlap, nulls = train_organ_clock(
            organ, logged, eligible, samples, scifo, out_dir, n_perm=n_perm
        )
        metric_rows.append(metrics)
        pred_frames.append(predictions)
        middle_rows.append(middle)
        overlap_rows.append(overlap)
        null_frames.append(nulls)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics.to_csv(clock_dir / "metrics.csv", index=False)
    predictions.to_csv(clock_dir / "predictions.csv", index=False)
    pd.DataFrame(middle_rows).to_csv(clock_dir / "middle_age.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(clock_dir / "scifo_overlap.csv", index=False)
    pd.concat(null_frames, ignore_index=True).to_csv(clock_dir / "permutation_null.csv", index=False)
    return predictions


def write_ageaccel(out_dir: Path, n_perm: int = 1_000) -> pd.DataFrame:
    """Pairwise residual correlations. Exploratory: each pair has few mice."""
    predictions = pd.read_csv(out_dir / "clocks" / "predictions.csv")
    predictions["ageaccel_l"] = np.nan
    for organ, sub in predictions.groupby("organ"):
        predictions.loc[sub.index, "ageaccel_l"] = ageaccel_l(
            sub["pred_loo"].to_numpy(dtype=float),
            sub["age_months"].to_numpy(dtype=float),
        )
    dest = out_dir / "ageaccel"
    dest.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(dest / "residuals.csv", index=False)
    organs = predictions["organ"].drop_duplicates().tolist()
    rng = np.random.default_rng(0)
    rows = []
    for i, left in enumerate(organs):
        for right in organs[i + 1 :]:
            a = predictions.loc[predictions["organ"] == left, ["mouse_id", "age_weeks", "ageaccel_l"]]
            b = predictions.loc[predictions["organ"] == right, ["mouse_id", "age_weeks", "ageaccel_l"]]
            merged = a.merge(b, on=["mouse_id", "age_weeks"], suffixes=("_a", "_b"))
            x = merged["ageaccel_l_a"].to_numpy(dtype=float)
            y = merged["ageaccel_l_b"].to_numpy(dtype=float)
            observed = _spearman(x, y) if len(merged) >= 5 else float("nan")
            count = 0
            finite = 0
            if np.isfinite(observed):
                for _ in range(n_perm):
                    permuted = y.copy()
                    for _, index in merged.groupby("age_weeks").groups.items():
                        positions = merged.index.get_indexer(index)
                        permuted[positions] = rng.permutation(permuted[positions])
                    rho = _spearman(x, permuted)
                    if np.isfinite(rho):
                        finite += 1
                        if abs(rho) >= abs(observed):
                            count += 1
            rows.append(
                {
                    "organ_a": left,
                    "organ_b": right,
                    "n_mice": int(len(merged)),
                    "spearman": observed,
                    "p_within_age": (count + 1) / (finite + 1) if finite else np.nan,
                    "note": "exploratory",
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(dest / "pairs.csv", index=False)
    return pairs
