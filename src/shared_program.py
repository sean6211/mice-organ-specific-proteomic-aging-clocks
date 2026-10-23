"""Shared-protein transfer clocks, coupled organ-intercept model, and tensor SVD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocess import abundance_block, preprocess_organ
from .train_clocks import ALPHAS, L1_RATIOS, RANDOM_STATE

ORGAN_ORDER = ["kidney", "spleen", "liver", "lung", "heart", "brain", "testis", "muscle"]


def _enet() -> Pipeline:
    return Pipeline(
        steps=[
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


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _organ_xy(matrix: pd.DataFrame, samples: pd.DataFrame, organ: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    processed = preprocess_organ(matrix)
    organ_samples = samples.loc[samples["organ"] == organ].copy()
    organ_samples["mouse_id"] = organ_samples["mouse_id"].astype(str)
    values = abundance_block(processed)
    cols = [
        mid
        for mid in organ_samples.sort_values(["age_months", "mouse_id"])["mouse_id"]
        if mid in values.columns
    ]
    x = values.reindex(columns=cols).T
    ages = organ_samples.set_index("mouse_id").loc[x.index, "age_months"].to_numpy(dtype=float)
    mice = x.index.astype(str).to_numpy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0))
    x.index = x.index.astype(str)
    return x, ages, mice


def shared_accessions(matrices: dict[str, pd.DataFrame], samples: pd.DataFrame) -> list[str]:
    shared: set[str] | None = None
    for organ, matrix in matrices.items():
        processed = preprocess_organ(matrix)
        accessions = set(abundance_block(processed).index.astype(str))
        shared = accessions if shared is None else shared & accessions
    if not shared:
        return []
    return sorted(shared)


def transfer_clocks(
    matrices: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    shared: list[str],
) -> pd.DataFrame:
    organs = [organ for organ in ORGAN_ORDER if organ in matrices]
    xy: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    native_loo: dict[str, dict[str, float]] = {}
    models_by_heldout: dict[str, dict[str, Pipeline]] = {}

    for organ in organs:
        x, y, mice = _organ_xy(matrices[organ], samples, organ)
        x = x.reindex(columns=shared)
        xy[organ] = (x, y, mice)
        oof = np.full(len(y), np.nan)
        model_map: dict[str, Pipeline] = {}
        print(f"  LOO transfer models: {organ}", flush=True)
        for train_idx, test_idx in LeaveOneOut().split(x):
            model = _enet()
            model.fit(x.iloc[train_idx], y[train_idx])
            oof[test_idx] = model.predict(x.iloc[test_idx])
            model_map[str(mice[test_idx[0]])] = model
        native_loo[organ] = {str(mouse): float(pred) for mouse, pred in zip(mice, oof)}
        models_by_heldout[organ] = model_map

    rows = []
    for source in organs:
        for target in organs:
            x_t, y_t, mice_t = xy[target]
            transferred = []
            chrono = []
            native = []
            for mouse, age in zip(mice_t, y_t):
                mouse = str(mouse)
                if mouse not in models_by_heldout[source] or mouse not in x_t.index:
                    continue
                pred = float(models_by_heldout[source][mouse].predict(x_t.loc[[mouse]])[0])
                transferred.append(pred)
                chrono.append(float(age))
                native.append(native_loo[target][mouse])
            transferred_a = np.asarray(transferred)
            chrono_a = np.asarray(chrono)
            native_a = np.asarray(native)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "n": int(len(chrono_a)),
                    "pearson_vs_age": _pearson(transferred_a, chrono_a),
                    "pearson_vs_target_loo": 1.0 if source == target else _pearson(transferred_a, native_a),
                    "mae_vs_age": float(mean_absolute_error(chrono_a, transferred_a))
                    if len(chrono_a)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def coupled_clock(
    matrices: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    shared: list[str],
) -> pd.DataFrame:
    organs = [organ for organ in ORGAN_ORDER if organ in matrices]
    xy = {}
    for organ in organs:
        x, y, mice = _organ_xy(matrices[organ], samples, organ)
        xy[organ] = (x.reindex(columns=shared), y, np.asarray(mice, dtype=str))

    all_mice = sorted({m for organ in organs for m in xy[organ][2]}, key=lambda m: int(m))
    loo_rows: list[dict[str, object]] = []

    for held in all_mice:
        x_train_blocks = []
        y_train = []
        organs_train = []
        intercepts: dict[str, float] = {}
        mus: dict[str, pd.Series] = {}
        sds: dict[str, pd.Series] = {}
        for organ in organs:
            x, y, mice = xy[organ]
            train = mice != held
            if train.sum() < 8:
                continue
            intercepts[organ] = float(np.mean(y[train]))
            mu = x.iloc[train].mean(axis=0)
            sd = x.iloc[train].std(axis=0).replace(0, 1.0).fillna(1.0)
            mus[organ] = mu
            sds[organ] = sd
            x_tr = (x.iloc[train] - mu) / sd
            x_train_blocks.append(x_tr.to_numpy(dtype=float))
            y_train.append(y[train] - intercepts[organ])
            organs_train.extend([organ] * int(train.sum()))
        if not x_train_blocks:
            continue
        x_train = np.vstack(x_train_blocks)
        y_train_a = np.concatenate(y_train)
        model = ElasticNetCV(
            l1_ratio=list(L1_RATIOS),
            alphas=ALPHAS,
            cv=5,
            max_iter=8000,
            tol=1e-3,
            fit_intercept=False,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train_s = scaler.fit_transform(x_train)
        model.fit(x_train_s, y_train_a)

        for organ in organs:
            x, y, mice = xy[organ]
            test = mice == held
            if not test.any() or organ not in intercepts:
                continue
            x_te = (x.iloc[test] - mus[organ]) / sds[organ]
            pred = float(model.predict(scaler.transform(x_te.to_numpy(dtype=float)))[0])
            loo_rows.append(
                {
                    "organ": organ,
                    "mouse_id": held,
                    "age_months": float(y[test][0]),
                    "pred_loo": pred + intercepts[organ],
                }
            )

    pred_frame = pd.DataFrame(loo_rows)
    rows = []
    for organ in organs:
        sub = pred_frame.loc[pred_frame["organ"] == organ]
        if sub.empty:
            continue
        rows.append(
            {
                "organ": organ,
                "model": "coupled_shared_weights",
                "n": int(len(sub)),
                "mae": float(mean_absolute_error(sub["age_months"], sub["pred_loo"])),
                "pearson_r": _pearson(sub["pred_loo"].to_numpy(), sub["age_months"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def tensor_svd(
    matrices: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    shared: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    organs = [organ for organ in ORGAN_ORDER if organ in matrices]
    xy = {organ: _organ_xy(matrices[organ], samples, organ) for organ in organs}
    mouse_sets = [set(map(str, xy[organ][2])) for organ in organs]
    common_mice = sorted(set.intersection(*mouse_sets), key=lambda m: int(m))
    ages = (
        samples.drop_duplicates("mouse_id")
        .assign(mouse_id=lambda d: d["mouse_id"].astype(str))
        .set_index("mouse_id")
        .loc[common_mice, "age_months"]
        .to_numpy(dtype=float)
    )
    slabs = []
    for organ in organs:
        x, _, mice = xy[organ]
        x = x.reindex(index=list(map(str, mice)), columns=shared)
        x = x.reindex(index=common_mice)
        x = x.fillna(x.median(axis=0))
        slabs.append(x.to_numpy(dtype=float))
    tensor = np.stack(slabs, axis=2)
    unfolded = tensor.reshape(tensor.shape[0], -1)
    unfolded = unfolded - unfolded.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(unfolded, full_matrices=False)
    mouse_pc = pd.DataFrame(
        {
            "mouse_id": common_mice,
            "age_months": ages,
            "pc1": u[:, 0] * s[0],
            "pc2": u[:, 1] * s[1],
        }
    )
    r = _pearson(mouse_pc["pc1"].to_numpy(), mouse_pc["age_months"].to_numpy())
    if r < 0:
        mouse_pc["pc1"] *= -1
        vt[0] *= -1
        r = -r
    mouse_pc["pc1_age_pearson"] = r
    organ_load = vt[0].reshape(len(shared), len(organs))
    organ_scores = pd.DataFrame(
        {
            "organ": organs,
            "loading_l2": np.sqrt((organ_load**2).sum(axis=0)),
            "mean_signed_loading": organ_load.mean(axis=0),
        }
    ).sort_values("loading_l2", ascending=False)
    return mouse_pc, organ_scores


def write_shared_program(
    matrices: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    out_dir: Path,
    metrics: pd.DataFrame | None = None,
) -> list[str]:
    shared_dir = out_dir / "shared_program"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared = shared_accessions(matrices, samples)
    pd.Series(shared, name="accession").to_csv(shared_dir / "shared_accessions.csv", index=False)
    print(f"Shared proteins in all eight clock matrices: {len(shared)}", flush=True)
    print("Transfer clocks ...", flush=True)
    transfer = transfer_clocks(matrices, samples, shared)
    transfer.to_csv(shared_dir / "transfer_clocks.csv", index=False)
    ks = transfer.loc[(transfer["source"] == "kidney") & (transfer["target"] == "spleen")]
    kb = transfer.loc[(transfer["source"] == "kidney") & (transfer["target"] == "brain")]
    if not ks.empty and not kb.empty:
        print(
            f"  kidney→spleen r(age)={ks.iloc[0]['pearson_vs_age']:.3f}; "
            f"kidney→brain r(age)={kb.iloc[0]['pearson_vs_age']:.3f}",
            flush=True,
        )
    print("Coupled shared-weight clock ...", flush=True)
    coupled = coupled_clock(matrices, samples, shared)
    if metrics is not None:
        private = metrics.loc[metrics["scheme"] == "loo", ["organ", "mae", "pearson_r"]].rename(
            columns={"mae": "private_loo_mae", "pearson_r": "private_loo_r"}
        )
        coupled = coupled.merge(private, on="organ", how="left")
        coupled["mae_ratio_vs_private"] = coupled["mae"] / coupled["private_loo_mae"]
    coupled.to_csv(shared_dir / "coupled_clock_metrics.csv", index=False)
    print("Unfolded tensor SVD ...", flush=True)
    mouse_pc, organ_scores = tensor_svd(matrices, samples, shared)
    mouse_pc.to_csv(shared_dir / "tensor_mouse_pc.csv", index=False)
    organ_scores.to_csv(shared_dir / "tensor_organ_loadings.csv", index=False)
    print(
        f"  tensor mouse-PC1 vs age Pearson r={float(mouse_pc['pc1_age_pearson'].iloc[0]):.3f}",
        flush=True,
    )
    return shared
