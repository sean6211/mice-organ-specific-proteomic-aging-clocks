"""Train one elastic-net chronological-age clock per organ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

from .ageaccel import ORGAN_ORDER
from .preprocess import abundance_block, preprocess_organ

L1_RATIOS = (0.2, 0.5, 0.8)
ALPHAS = np.logspace(-2, 1, 10)
RANDOM_STATE = 0


def _pipeline() -> Pipeline:
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
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _xy(matrix: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    values = abundance_block(matrix)
    sample_order = samples.sort_values(["age_months", "mouse_id"])
    cols = [str(m) for m in sample_order["mouse_id"].tolist()]
    x = values.reindex(columns=cols).T
    y = sample_order["age_months"].to_numpy(dtype=float)
    mice = sample_order["mouse_id"].to_numpy()
    return x, y, mice


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": float(mae),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        "pearson_r": pearson,
    }


def _cross_val_predict(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    scheme: str,
    collect_coefs: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    preds = np.full(len(y), np.nan)
    coef_folds: list[np.ndarray] = []
    if scheme == "loao":
        splitter = LeaveOneGroupOut().split(x, y, groups)
    elif scheme == "loo":
        splitter = LeaveOneOut().split(x, y)
    else:
        raise ValueError(scheme)
    for train_idx, test_idx in splitter:
        model = _pipeline()
        model.fit(x.iloc[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x.iloc[test_idx])
        if collect_coefs:
            coef_folds.append(model.named_steps["model"].coef_.copy())
    stability = None
    if collect_coefs:
        coef_mat = np.vstack(coef_folds)
        selected = np.abs(coef_mat) > 1e-12
        stability = pd.DataFrame(
            {
                "accession": x.columns,
                "selection_frequency": selected.mean(axis=0),
                "mean_coef": np.where(selected.any(axis=0), coef_mat.sum(axis=0) / np.maximum(selected.sum(axis=0), 1), 0.0),
                "n_folds": coef_mat.shape[0],
            }
        )
        stability = stability.sort_values("selection_frequency", ascending=False)
    return preds, stability


def train_organ(
    organ: str,
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = preprocess_organ(matrix)
    organ_samples = samples.loc[samples["organ"] == organ].copy()
    x, y, mice = _xy(processed, organ_samples)
    loao_pred, _ = _cross_val_predict(x, y, groups=y, scheme="loao")
    loo_pred, stability = _cross_val_predict(x, y, groups=None, scheme="loo", collect_coefs=True)
    if stability is None:
        raise RuntimeError("LOO stability table was not collected")
    stability.insert(0, "organ", organ)
    if "description" in processed.columns:
        stability["description"] = processed["description"].reindex(stability["accession"]).to_numpy()

    full_model = _pipeline()
    full_model.fit(x, y)
    if out_dir is not None:
        write_full_fit_pipeline(organ, full_model, x, out_dir)
    coef = full_model.named_steps["model"].coef_
    descriptions = processed["description"] if "description" in processed.columns else None
    coef_frame = pd.DataFrame(
        {
            "organ": organ,
            "accession": x.columns,
            "coefficient": coef,
            "description": descriptions.reindex(x.columns).to_numpy() if descriptions is not None else None,
        }
    )
    coef_frame = coef_frame.loc[coef_frame["coefficient"].abs() > 1e-12].sort_values(
        "coefficient", key=lambda s: s.abs(), ascending=False
    )

    pred_frame = pd.DataFrame(
        {
            "organ": organ,
            "mouse_id": mice,
            "age_months": y,
            "pred_loao": loao_pred,
            "pred_loo": loo_pred,
            "gap_loao": loao_pred - y,
            "gap_loo": loo_pred - y,
        }
    )
    metrics = pd.DataFrame(
        [
            {"organ": organ, "scheme": "loao", "n_proteins": x.shape[1], **_metrics(y, loao_pred)},
            {"organ": organ, "scheme": "loo", "n_proteins": x.shape[1], **_metrics(y, loo_pred)},
        ]
    )
    return metrics, pred_frame, coef_frame, stability


def train_all(
    matrices: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clock_dir = out_dir / "clocks"
    coef_dir = clock_dir / "coefficients"
    stab_dir = clock_dir / "stability"
    coef_dir.mkdir(parents=True, exist_ok=True)
    stab_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    pred_frames = []
    for organ, matrix in matrices.items():
        print(f"Training {organ} clock ({matrix.shape[0]} proteins before filter) ...", flush=True)
        metrics, preds, coefs, stability = train_organ(organ, matrix, samples, out_dir=out_dir)
        metric_frames.append(metrics)
        pred_frames.append(preds)
        coefs.to_csv(coef_dir / f"{organ}.csv", index=False)
        stability.to_csv(stab_dir / f"{organ}.csv", index=False)
        n_stable = int((stability["selection_frequency"] > 0.5).sum())
        print(
            f"  {organ}: {int(metrics.iloc[0]['n_proteins'])} proteins, "
            f"LOAO MAE={metrics.iloc[0]['mae']:.2f} mo, "
            f"LOO MAE={metrics.iloc[1]['mae']:.2f} mo, "
            f"{len(coefs)} non-zero weights, {n_stable} stable (>50% LOO)",
            flush=True,
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics.to_csv(clock_dir / "metrics.csv", index=False)
    predictions.to_csv(clock_dir / "predictions.csv", index=False)
    return metrics, predictions


def write_full_fit_pipeline(organ: str, model, x: pd.DataFrame, out_dir: Path) -> Path:
    """Save intercept, imputer medians, and scaler mean/SD for the full-fit clock."""
    dest = out_dir / "clocks" / "pipelines"
    dest.mkdir(parents=True, exist_ok=True)
    imputer = model.named_steps["impute"]
    scaler = model.named_steps["scale"]
    elastic = model.named_steps["model"]
    features = pd.DataFrame(
        {
            "organ": organ,
            "accession": x.columns.astype(str),
            "imputer_median": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "coefficient": elastic.coef_,
        }
    )
    features.to_csv(dest / f"{organ}_features.csv", index=False)
    dump(model, dest / f"{organ}_pipeline.joblib")
    intercepts_path = dest / "intercepts.csv"
    row = pd.DataFrame(
        [
            {
                "organ": organ,
                "intercept": float(elastic.intercept_),
                "n_features": int(x.shape[1]),
                "n_nonzero": int(np.sum(np.abs(elastic.coef_) > 1e-12)),
                "n_samples": int(x.shape[0]),
            }
        ]
    )
    if intercepts_path.exists():
        prev = pd.read_csv(intercepts_path)
        prev = prev.loc[prev["organ"] != organ]
        row = pd.concat([prev, row], ignore_index=True)
    row = row.sort_values("organ").reset_index(drop=True)
    row.to_csv(intercepts_path, index=False)
    return dest


def load_full_fit_pipeline(out_dir: Path, organ: str) -> tuple[pd.DataFrame, float]:
    dest = out_dir / "clocks" / "pipelines"
    features = pd.read_csv(dest / f"{organ}_features.csv")
    features["accession"] = features["accession"].astype(str)
    intercepts = pd.read_csv(dest / "intercepts.csv")
    hit = intercepts.loc[intercepts["organ"] == organ]
    if hit.empty:
        raise FileNotFoundError(f"No intercept for {organ}")
    return features, float(hit["intercept"].iloc[0])


def predict_months(values: pd.DataFrame, features: pd.DataFrame, intercept: float) -> tuple[pd.Series, pd.DataFrame]:
    """Apply saved Scifo full-fit pipeline. Missing proteins get the training imputer median."""
    block = values.drop(columns=["description"], errors="ignore")
    block.index = block.index.map(str)
    block.columns = block.columns.map(str)
    feats = features.drop_duplicates("accession").set_index("accession")
    aligned = block.reindex(feats.index)
    n_observed = aligned.notna().sum(axis=0)
    nonzero_idx = feats.loc[feats["coefficient"].abs() > 1e-12].index
    n_obs_nonzero = aligned.reindex(nonzero_idx).notna().sum(axis=0)
    filled = aligned.T.apply(pd.to_numeric, errors="coerce")
    filled = filled.fillna(feats["imputer_median"])
    scale = feats["scaler_scale"].replace(0, np.nan)
    scaled = filled.sub(feats["scaler_mean"], axis=1).div(scale, axis=1)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pred = scaled.dot(feats["coefficient"]) + intercept
    pred.name = "pred_months"
    coverage = pd.DataFrame(
        {
            "sample_id": pred.index.astype(str),
            "n_features": int(len(feats)),
            "n_observed": n_observed.reindex(pred.index).to_numpy(),
            "n_nonzero": int((feats["coefficient"].abs() > 1e-12).sum()),
            "n_nonzero_observed": n_obs_nonzero.reindex(pred.index).to_numpy(),
        }
    )
    return pred, coverage


def median_center(block: pd.DataFrame) -> pd.DataFrame:
    values = block.drop(columns=["description"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    return values.sub(values.median(axis=0, skipna=True), axis=1)


def fit_and_save_pipelines(out_dir: Path) -> pd.DataFrame:
    """Re-fit full-fit ElasticNet only (no LOO) and write intercept/scaler/imputer."""
    samples = pd.read_csv(out_dir / "metadata" / "samples.csv")
    matrix_dir = out_dir / "matrices"
    rows = []
    for organ in ORGAN_ORDER:
        path = matrix_dir / f"{organ}_lfq.csv"
        if not path.exists():
            continue
        print(f"  full-fit pipeline {organ} ...", flush=True)
        matrix = pd.read_csv(path, index_col=0)
        processed = preprocess_organ(matrix)
        organ_samples = samples.loc[samples["organ"] == organ].copy()
        x, y, _mice = _xy(processed, organ_samples)
        model = _pipeline()
        model.fit(x, y)
        write_full_fit_pipeline(organ, model, x, out_dir)
        pred = model.predict(x)
        features, intercept = load_full_fit_pipeline(out_dir, organ)
        pred_csv, _cov = predict_months(x.T, features, intercept)
        csv_mae = float(np.mean(np.abs(pred - pred_csv.reindex(x.index).to_numpy())))
        saved = pd.read_csv(out_dir / "clocks" / "coefficients" / f"{organ}.csv")
        saved_w = saved.set_index(saved["accession"].astype(str))["coefficient"]
        new_w = pd.Series(model.named_steps["model"].coef_, index=x.columns.astype(str))
        shared = saved_w.index.intersection(new_w.index)
        coef_mae = float(np.mean(np.abs(saved_w.reindex(shared).fillna(0) - new_w.reindex(shared).fillna(0))))
        rows.append(
            {
                "organ": organ,
                "n_features": int(x.shape[1]),
                "n_nonzero": int(np.sum(np.abs(new_w) > 1e-12)),
                "intercept": float(model.named_steps["model"].intercept_),
                "in_sample_mae": float(mean_absolute_error(y, pred)),
                "in_sample_r": float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-12 else np.nan,
                "coef_mae_vs_saved": coef_mae,
                "csv_vs_sklearn_mae": csv_mae,
            }
        )
        print(
            f"    intercept={model.named_steps['model'].intercept_:.3f}  "
            f"in-sample MAE={mean_absolute_error(y, pred):.2f} mo  "
            f"coef MAE vs saved={coef_mae:.4g}  "
            f"CSV vs sklearn={csv_mae:.3g}",
            flush=True,
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "clocks" / "pipelines" / "fit_summary.csv", index=False)
    return summary
