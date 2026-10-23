"""Explainability of stably selected organ-clock proteins. Frozen weights only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact, spearmanr

from .ageaccel import ORGAN_ORDER
from .annotate_clocks import STABLE_CUTOFF, _gprofiler_profile
from .preprocess import abundance_block, preprocess_organ

N_REDUNDANCY = 500
TOP_N = 15

KEYWORD_PATTERNS: dict[str, str] = {
    "mitochondrion": r"mitochondri",
    "ribosome": r"ribosom|40S ribosomal|60S ribosomal",
    "oxphos": r"cytochrome|ATP synthase|NADH dehydrogenase|ubiquinol|succinate dehydrogenase",
    "immunoglobulin": r"immunoglobulin|\bIg |Ig gamma|Ig kappa|Ig mu|Ig alpha|Ig delta",
    "collagen_ecm": r"collagen|laminin|fibronectin|elastin|extracellular matrix",
    "proteasome": r"proteasome",
    "ubiquitin": r"ubiquitin",
    "histone": r"\bhistone\b",
    "chaperone": r"heat shock|chaperon",
    "complement": r"complement ",
    "myosin_actin": r"myosin|tropomyosin|troponin|\bactin\b",
    "kinase": r"kinase",
    "transporter": r"transporter|solute carrier|ABC transporter",
    "transcription": r"transcription factor|RNA polymerase",
}

UNIPROT_KEYWORD_CLASSES: dict[str, tuple[str, ...]] = {
    "mitochondrion": ("mitochondrion",),
    "secreted": ("secreted",),
    "nucleus": ("nucleus",),
    "membrane": ("membrane",),
    "cytoplasm": ("cytoplasm",),
    "endoplasmic reticulum": ("endoplasmic reticulum",),
    "lysosome": ("lysosome",),
    "ribosome": ("ribosome",),
    "oxidoreductase": ("oxidoreductase",),
    "immunity": ("immunity", "innate immunity"),
    "transcription": ("transcription",),
    "kinase": ("kinase",),
    "chaperone": ("chaperone",),
}


def _short_name(description: str) -> str:
    text = str(description)
    if text.lower() in ("nan", "none"):
        return ""
    return text.split("[OS=", 1)[0].strip().rstrip(",")


def _load_processed(out_dir: Path, organ: str, samples: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(out_dir / "matrices" / f"{organ}_lfq.csv", index_col=0)
    processed = preprocess_organ(raw)
    values = abundance_block(processed)
    organ_samples = samples.loc[samples["organ"] == organ].copy()
    organ_samples["mouse_id"] = organ_samples["mouse_id"].astype(str)
    cols = [mid for mid in organ_samples["mouse_id"].tolist() if mid in values.columns]
    values = values.reindex(columns=cols)
    ages = organ_samples.set_index("mouse_id").loc[cols, "age_months"].to_numpy(dtype=float)
    return values, ages


def _zscore(mat: np.ndarray) -> np.ndarray:
    med = np.nanmedian(mat, axis=0, keepdims=True)
    filled = np.where(np.isfinite(mat), mat, med)
    mu = filled.mean(axis=0, keepdims=True)
    sd = filled.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (filled - mu) / sd


def _spearman_rows(values: pd.DataFrame, ages: np.ndarray) -> pd.DataFrame:
    rows = []
    for acc in values.index.astype(str):
        y = values.loc[acc].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(ages)
        if mask.sum() < 6:
            rows.append({"accession": acc, "spearman_vs_age": np.nan, "p_parametric": np.nan, "n_mice": int(mask.sum())})
            continue
        rho, p_val = spearmanr(y[mask], ages[mask])
        rows.append(
            {
                "accession": acc,
                "spearman_vs_age": float(rho) if np.isfinite(rho) else np.nan,
                "p_parametric": float(p_val) if np.isfinite(p_val) else np.nan,
                "n_mice": int(mask.sum()),
            }
        )
    out = pd.DataFrame(rows)
    finite = out["p_parametric"].to_numpy(dtype=float)
    q = np.full(len(out), np.nan)
    ok = np.isfinite(finite)
    if ok.any():
        q[ok] = false_discovery_control(finite[ok], method="bh")
    out["q_bh"] = q
    return out


def organ_features(out_dir: Path, organ: str, samples: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    stability = pd.read_csv(out_dir / "clocks" / "stability" / f"{organ}.csv")
    stability["accession"] = stability["accession"].astype(str)
    coef = pd.read_csv(out_dir / "clocks" / "coefficients" / f"{organ}.csv")
    coef["accession"] = coef["accession"].astype(str)
    values, ages = _load_processed(out_dir, organ, samples)
    uni = _spearman_rows(values, ages)
    mem = membership.loc[membership["organ"] == organ].copy()
    mem["accession"] = mem["accession"].astype(str)

    secreted_path = out_dir / "biology" / "stable_secreted.csv"
    secreted = pd.DataFrame()
    if secreted_path.exists():
        secreted = pd.read_csv(secreted_path)
        secreted = secreted.loc[secreted["organ"] == organ].copy()
        secreted["accession"] = secreted["accession"].astype(str)

    stable = stability.loc[stability["selection_frequency"] > STABLE_CUTOFF].copy()
    keep = [acc for acc in stable["accession"] if acc in values.index]
    sub = values.loc[keep]
    xz = _zscore(sub.to_numpy(dtype=float).T)
    w = stable.set_index("accession").loc[keep, "mean_coef"].to_numpy(dtype=float)
    score = xz @ w
    rho_full, _ = spearmanr(score, ages)
    mean_abs_contrib = np.mean(np.abs(xz * w[None, :]), axis=0)
    loo_delta = np.empty(len(keep), dtype=float)
    for j in range(len(keep)):
        score_j = score - xz[:, j] * w[j]
        rho_j, _ = spearmanr(score_j, ages)
        loo_delta[j] = float(rho_full - rho_j) if np.isfinite(rho_full) and np.isfinite(rho_j) else np.nan

    feat = stable.merge(uni, on="accession", how="left")
    feat = feat.merge(coef[["accession", "coefficient"]].rename(columns={"coefficient": "full_fit_coef"}), on="accession", how="left")
    feat = feat.merge(mem[["accession", "age_de", "full_clock"]], on="accession", how="left")
    if not secreted.empty:
        feat = feat.merge(
            secreted[["accession", "secreted", "has_signal_peptide", "circulating_reporter_candidate", "uniprot_keywords"]],
            on="accession",
            how="left",
        )
    else:
        feat["secreted"] = False
        feat["has_signal_peptide"] = False
        feat["circulating_reporter_candidate"] = False
        feat["uniprot_keywords"] = ""
    contrib_map = {acc: float(v) for acc, v in zip(keep, mean_abs_contrib)}
    delta_map = {acc: float(v) for acc, v in zip(keep, loo_delta)}
    feat["mean_abs_contribution"] = feat["accession"].map(contrib_map)
    feat["loo_delta_spearman"] = feat["accession"].map(delta_map)
    feat["sign_coef"] = np.sign(feat["mean_coef"].to_numpy(dtype=float))
    feat["sign_univariate"] = np.sign(feat["spearman_vs_age"].to_numpy(dtype=float))
    feat["sign_concordant"] = feat["sign_coef"] * feat["sign_univariate"] > 0
    feat["short_name"] = feat["description"].map(_short_name)
    feat["frozen_score_spearman"] = float(rho_full) if np.isfinite(rho_full) else np.nan
    feat["n_stable"] = int(len(feat))
    feat["organ"] = organ
    feat["abs_mean_coef"] = feat["mean_coef"].abs()
    return feat.sort_values(["selection_frequency", "abs_mean_coef"], ascending=False)


def organ_summary(feat: pd.DataFrame) -> dict:
    n = int(len(feat))
    always = int((feat["selection_frequency"] >= 1.0 - 1e-12).sum())
    high = int((feat["selection_frequency"] > 0.8).sum())
    concord = feat.loc[feat["sign_coef"] != 0]
    concord = concord.loc[concord["sign_univariate"] != 0]
    n_conc = int(concord["sign_concordant"].sum()) if len(concord) else 0
    n_signed = int(len(concord))
    de = feat["age_de"].fillna(False).astype(bool)
    secreted = feat["secreted"].fillna(False).astype(bool)
    return {
        "organ": feat["organ"].iloc[0],
        "n_stable": n,
        "n_always_selected": always,
        "n_freq_gt_0.8": high,
        "median_frequency": float(feat["selection_frequency"].median()),
        "median_abs_mean_coef": float(feat["mean_coef"].abs().median()),
        "median_abs_spearman": float(feat["spearman_vs_age"].abs().median()),
        "n_univariate_q_lt_0.05": int((feat["q_bh"] < 0.05).sum()),
        "frac_univariate_q_lt_0.05": float((feat["q_bh"] < 0.05).mean()),
        "n_sign_concordant": n_conc,
        "n_signed": n_signed,
        "frac_sign_concordant": float(n_conc / n_signed) if n_signed else np.nan,
        "n_age_de": int(de.sum()),
        "frac_age_de": float(de.mean()),
        "n_secreted": int(secreted.sum()),
        "frac_secreted": float(secreted.mean()),
        "frozen_score_spearman": float(feat["frozen_score_spearman"].iloc[0]),
        "sum_loo_delta": float(feat["loo_delta_spearman"].sum()),
        "max_loo_delta": float(feat["loo_delta_spearman"].max()),
        "top_loo_protein": str(feat.loc[feat["loo_delta_spearman"].idxmax(), "accession"]) if feat["loo_delta_spearman"].notna().any() else "",
        "top_loo_name": str(feat.loc[feat["loo_delta_spearman"].idxmax(), "short_name"]) if feat["loo_delta_spearman"].notna().any() else "",
    }


def keyword_enrichment(out_dir: Path) -> pd.DataFrame:
    rows = []
    for organ in ORGAN_ORDER:
        stab = pd.read_csv(out_dir / "clocks" / "stability" / f"{organ}.csv")
        stab["accession"] = stab["accession"].astype(str)
        stab["description"] = stab["description"].astype(str)
        stable = stab["selection_frequency"] > STABLE_CUTOFF
        n_stable = int(stable.sum())
        n_bg = int(len(stab))
        for name, pattern in KEYWORD_PATTERNS.items():
            hit = stab["description"].str.contains(pattern, case=False, regex=True, na=False)
            a = int((hit & stable).sum())
            b = int((~hit & stable).sum())
            c = int((hit & ~stable).sum())
            d = int((~hit & ~stable).sum())
            table = np.array([[a, b], [c, d]], dtype=int)
            if table.min() < 0 or table.sum() == 0:
                oddsr, p_val = np.nan, np.nan
            else:
                oddsr, p_val = fisher_exact(table, alternative="two-sided")
            rows.append(
                {
                    "organ": organ,
                    "keyword": name,
                    "n_stable_hit": a,
                    "n_stable": n_stable,
                    "n_background_hit": int(hit.sum()),
                    "n_background": n_bg,
                    "frac_stable": a / n_stable if n_stable else np.nan,
                    "frac_background": float(hit.mean()),
                    "odds_ratio": float(oddsr) if np.isfinite(oddsr) else np.nan,
                    "p_fisher": float(p_val) if np.isfinite(p_val) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    q = np.full(len(out), np.nan)
    ok = np.isfinite(out["p_fisher"].to_numpy(dtype=float))
    if ok.any():
        q[ok] = false_discovery_control(out.loc[ok, "p_fisher"].to_numpy(dtype=float), method="bh")
    out["q_bh"] = q
    return out


def uniprot_composition(feat_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for organ, sub in feat_all.groupby("organ"):
        kw = sub["uniprot_keywords"].fillna("").astype(str).str.lower()
        n = int(len(sub))
        for name, needles in UNIPROT_KEYWORD_CLASSES.items():
            hit = np.zeros(n, dtype=bool)
            for needle in needles:
                hit |= kw.str.contains(needle, regex=False)
            rows.append(
                {
                    "organ": organ,
                    "uniprot_class": name,
                    "n": int(hit.sum()),
                    "n_stable": n,
                    "fraction": float(hit.mean()),
                }
            )
    return pd.DataFrame(rows)


def cross_organ_overlap(feat_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_organ = {
        organ: set(sub["accession"].astype(str))
        for organ, sub in feat_all.groupby("organ")
    }
    sign = feat_all.set_index(["organ", "accession"])["sign_coef"]
    pair_rows = []
    organs = [o for o in ORGAN_ORDER if o in by_organ]
    for i, a in enumerate(organs):
        for b in organs[i + 1 :]:
            shared = by_organ[a] & by_organ[b]
            n_shared = len(shared)
            agree = 0
            for acc in shared:
                sa = float(sign.loc[(a, acc)])
                sb = float(sign.loc[(b, acc)])
                if sa * sb > 0:
                    agree += 1
            union = len(by_organ[a] | by_organ[b])
            pair_rows.append(
                {
                    "organ_a": a,
                    "organ_b": b,
                    "n_a": len(by_organ[a]),
                    "n_b": len(by_organ[b]),
                    "n_shared": n_shared,
                    "jaccard": n_shared / union if union else np.nan,
                    "n_sign_agree": agree,
                    "frac_sign_agree": agree / n_shared if n_shared else np.nan,
                }
            )
    counts = feat_all.groupby("accession").agg(
        n_organs=("organ", "nunique"),
        organs=("organ", lambda s: ",".join(sorted(s))),
        short_name=("short_name", "first"),
        description=("description", "first"),
    ).reset_index()
    multi = counts.loc[counts["n_organs"] >= 2].sort_values(["n_organs", "accession"], ascending=[False, True])
    return pd.DataFrame(pair_rows), multi


def redundancy(out_dir: Path, samples: pd.DataFrame, n_perm: int = N_REDUNDANCY) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for organ in ORGAN_ORDER:
        values, _ages = _load_processed(out_dir, organ, samples)
        stab = pd.read_csv(out_dir / "clocks" / "stability" / f"{organ}.csv")
        keep = [a for a in stab.loc[stab["selection_frequency"] > STABLE_CUTOFF, "accession"].astype(str) if a in values.index]
        if len(keep) < 3:
            continue
        ranked = _rank_rows(values.to_numpy(dtype=float))
        idx = {acc: i for i, acc in enumerate(values.index.astype(str))}
        keep_i = np.array([idx[a] for a in keep], dtype=int)
        obs = _median_abs_offdiag(np.corrcoef(ranked[keep_i]))
        n_bg = ranked.shape[0]
        k = len(keep)
        nulls = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            draw = rng.choice(n_bg, size=k, replace=False)
            nulls[i] = _median_abs_offdiag(np.corrcoef(ranked[draw]))
        p = float((np.sum(nulls >= obs) + 1) / (n_perm + 1))
        rows.append(
            {
                "organ": organ,
                "n_stable": k,
                "median_abs_pairwise_spearman": obs,
                "null_mean": float(np.mean(nulls)),
                "null_p95": float(np.quantile(nulls, 0.95)),
                "p_permutation": p,
                "n_perm": n_perm,
            }
        )
    return pd.DataFrame(rows)


def _rank_rows(mat: np.ndarray) -> np.ndarray:
    filled = np.array(mat, dtype=float, copy=True)
    for i in range(filled.shape[0]):
        row = filled[i]
        med = np.nanmedian(row)
        row = np.where(np.isfinite(row), row, med)
        filled[i] = pd.Series(row).rank(method="average").to_numpy(dtype=float)
    return filled


def _median_abs_offdiag(corr: np.ndarray) -> float:
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    tri = np.abs(corr[np.triu_indices(n, k=1)])
    tri = tri[np.isfinite(tri)]
    return float(np.median(tri)) if len(tri) else float("nan")


def gprofiler_vs_proteome(out_dir: Path) -> pd.DataFrame:
    frames = []
    dest = out_dir / "clock_explain" / "gprofiler"
    dest.mkdir(parents=True, exist_ok=True)
    for organ in ORGAN_ORDER:
        stab = pd.read_csv(out_dir / "clocks" / "stability" / f"{organ}.csv")
        query = stab.loc[stab["selection_frequency"] > STABLE_CUTOFF, "accession"].astype(str).tolist()
        background = stab["accession"].astype(str).tolist()
        cache = dest / f"{organ}_stable_vs_proteome.csv"
        result = _gprofiler_profile(query, background, cache)
        if result is None or result.empty or "error" in result.columns:
            continue
        result = result.copy()
        result.insert(0, "organ", organ)
        frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_clock_explain(out_dir: Path) -> dict[str, pd.DataFrame]:
    dest = out_dir / "clock_explain"
    dest.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(out_dir / "metadata" / "samples.csv")
    membership = pd.read_csv(out_dir / "de_overlap" / "set_membership.csv")
    feats = []
    print("  per-organ stable features ...", flush=True)
    for organ in ORGAN_ORDER:
        print(f"    {organ}", flush=True)
        feats.append(organ_features(out_dir, organ, samples, membership))
    feat_all = pd.concat(feats, ignore_index=True)
    summaries = pd.DataFrame([organ_summary(f) for f in feats])
    print("  description-keyword enrichment ...", flush=True)
    keywords = keyword_enrichment(out_dir)
    composition = uniprot_composition(feat_all)
    pairs, multi = cross_organ_overlap(feat_all)
    print("  coexpression redundancy vs random proteins ...", flush=True)
    redun = redundancy(out_dir, samples)
    print("  g:Profiler stable vs organ proteome ...", flush=True)
    gp = gprofiler_vs_proteome(out_dir)
    top = (
        feat_all.sort_values(["organ", "mean_abs_contribution", "selection_frequency"], ascending=[True, False, False])
        .groupby("organ", sort=False)
        .head(TOP_N)
    )
    feat_all.to_csv(dest / "stable_features.csv", index=False)
    summaries.to_csv(dest / "organ_summary.csv", index=False)
    keywords.to_csv(dest / "keyword_enrichment.csv", index=False)
    composition.to_csv(dest / "uniprot_composition.csv", index=False)
    pairs.to_csv(dest / "cross_organ_overlap.csv", index=False)
    multi.to_csv(dest / "proteins_in_multiple_clocks.csv", index=False)
    redun.to_csv(dest / "redundancy.csv", index=False)
    top.to_csv(dest / "top_proteins.csv", index=False)
    if not gp.empty:
        gp.to_csv(dest / "gprofiler_stable_vs_proteome.csv", index=False)
    return {
        "features": feat_all,
        "summary": summaries,
        "keywords": keywords,
        "uniprot": composition,
        "overlap": pairs,
        "multi": multi,
        "redundancy": redun,
        "top": top,
        "gprofiler": gp,
    }
