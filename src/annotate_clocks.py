"""Annotate stable-clock vs age-DE protein sets (g:Profiler + UniProt secreted)."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .ageaccel import ORGAN_ORDER

STABLE_CUTOFF = 0.5
STABLE_STRICT = 0.8
GPROFILER_SOURCES = ["GO:BP", "GO:CC", "REAC", "KEGG"]
UNIPROT_CHUNK = 50
EXTRACELLULAR_NAME = r"extracellular|secreted|external side of plasma"


def _load_stability(stab_dir: Path, organ: str) -> pd.DataFrame:
    frame = pd.read_csv(stab_dir / f"{organ}.csv")
    frame["accession"] = frame["accession"].astype(str)
    return frame


def protein_sets(
    membership: pd.DataFrame,
    stability: pd.DataFrame,
    de: pd.DataFrame,
    cutoff: float,
) -> dict[str, pd.DataFrame]:
    mem = membership.copy()
    mem["accession"] = mem["accession"].astype(str)
    de_ids = set(mem.loc[mem["age_de"].astype(bool), "accession"])
    if np.isclose(cutoff, STABLE_CUTOFF) and "stable_clock" in mem.columns:
        stab_ids = set(mem.loc[mem["stable_clock"].astype(bool), "accession"])
    else:
        stab_ids = set(stability.loc[stability["selection_frequency"] > cutoff, "accession"].astype(str))
    rows = []
    for acc in sorted(de_ids | stab_ids):
        is_de = acc in de_ids
        is_st = acc in stab_ids
        if is_de and is_st:
            set_name = "intersection"
        elif is_st:
            set_name = "stable_only"
        else:
            set_name = "de_only"
        rows.append({"accession": acc, "set": set_name, "age_de": is_de, "stable_clock": is_st})
    sets = pd.DataFrame(rows)
    stab_cols = stability[["accession", "mean_coef", "selection_frequency", "description"]].drop_duplicates("accession")
    sets = sets.merge(stab_cols, on="accession", how="left")
    if "description" in de.columns:
        de_desc = de[["accession", "description"]].drop_duplicates("accession")
        de_desc["accession"] = de_desc["accession"].astype(str)
        sets = sets.merge(de_desc, on="accession", how="left", suffixes=("", "_de"))
        if "description_de" in sets.columns:
            sets["description"] = sets["description"].fillna(sets["description_de"])
            sets = sets.drop(columns=["description_de"])
    sets["coef_sign"] = np.where(
        sets["mean_coef"].isna(),
        "",
        np.where(sets["mean_coef"] > 0, "positive", np.where(sets["mean_coef"] < 0, "negative", "zero")),
    )
    return {
        "all": sets,
        "de_only": sets.loc[sets["set"] == "de_only"].copy(),
        "stable_only": sets.loc[sets["set"] == "stable_only"].copy(),
        "intersection": sets.loc[sets["set"] == "intersection"].copy(),
    }


def _write_cache(table: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    json_path = cache_path.with_suffix(".json")
    json_path.write_text(table.to_json(orient="records"), encoding="utf-8")


def _gprofiler_profile(query: list[str], background: list[str], cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    if len(query) < 3:
        empty = pd.DataFrame()
        _write_cache(empty, cache_path)
        return empty
    try:
        from gprofiler import GProfiler
    except ImportError as exc:
        err = pd.DataFrame([{"error": f"gprofiler-official not installed: {exc}"}])
        _write_cache(err, cache_path)
        return err
    gp = GProfiler(return_dataframe=True)
    try:
        result = gp.profile(
            organism="mmusculus",
            query=query,
            background=background,
            sources=GPROFILER_SOURCES,
            user_threshold=0.05,
            significance_threshold_method="fdr",
            domain_scope="custom",
            no_evidences=False,
        )
    except Exception as exc:  # noqa: BLE001
        err = pd.DataFrame([{"error": str(exc)}])
        _write_cache(err, cache_path)
        return err
    if result is None or result.empty:
        result = pd.DataFrame()
    _write_cache(result, cache_path)
    return result


def enrich_sets(
    organ: str,
    sets: dict[str, pd.DataFrame],
    background: list[str],
    cache_dir: Path,
) -> pd.DataFrame:
    frames = []
    for set_name in ("de_only", "stable_only", "intersection"):
        query = sets[set_name]["accession"].astype(str).tolist()
        cache = cache_dir / f"{organ}_{set_name}.csv"
        table = _gprofiler_profile(query, background, cache)
        if table.empty:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "organ": organ,
                            "set": set_name,
                            "n_query": len(query),
                            "source": "",
                            "native": "",
                            "name": "",
                            "p_value": np.nan,
                            "term_size": np.nan,
                            "query_size": len(query),
                            "intersection_size": 0,
                            "note": "no_terms_or_failed",
                        }
                    ]
                )
            )
            continue
        if "error" in table.columns:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "organ": organ,
                            "set": set_name,
                            "n_query": len(query),
                            "source": "",
                            "native": "",
                            "name": str(table["error"].iloc[0]),
                            "p_value": np.nan,
                            "note": "gprofiler_error",
                        }
                    ]
                )
            )
            continue
        keep = table.copy()
        keep.insert(0, "organ", organ)
        keep.insert(1, "set", set_name)
        keep.insert(2, "n_query", len(query))
        frames.append(keep)
    return pd.concat(frames, ignore_index=True)


def _uniprot_chunk(accessions: list[str]) -> pd.DataFrame:
    encoded = ",".join(accessions)
    fields = "accession,keyword,cc_subcellular_location,ft_signal,protein_name,cc_function"
    params = urllib.parse.urlencode(
        {
            "accessions": encoded,
            "fields": fields,
            "format": "tsv",
        }
    )
    url = f"https://rest.uniprot.org/uniprotkb/accessions?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mice_Proteomic/biology"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        text = resp.read().decode("utf-8")
    if not text.strip() or text.lstrip().startswith("{"):
        return pd.DataFrame()
    return pd.read_csv(StringIO(text), sep="\t")


def fetch_uniprot_secreted(accessions: list[str], cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    unique = sorted({str(a) for a in accessions if str(a) and str(a) != "nan"})
    chunks = []
    for i in range(0, len(unique), UNIPROT_CHUNK):
        batch = unique[i : i + UNIPROT_CHUNK]
        try:
            chunks.append(_uniprot_chunk(batch))
            time.sleep(0.25)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            chunks.append(pd.DataFrame([{"Entry": "", "error": str(exc)}]))
    if not chunks:
        out = pd.DataFrame()
        out.to_csv(cache_path, index=False)
        return out
    raw = pd.concat(chunks, ignore_index=True)
    raw.to_csv(cache_path, index=False)
    return raw


def _col(frame: pd.DataFrame, *needles: str) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for needle in needles:
        for key, orig in lower.items():
            if needle in key:
                return orig
    return None


def _is_secreted(row: pd.Series, kw_col: str | None, loc_col: str | None, sig_col: str | None) -> tuple[bool, bool]:
    kw = str(row.get(kw_col, "") if kw_col else "").lower()
    loc = str(row.get(loc_col, "") if loc_col else "").lower()
    sig = str(row.get(sig_col, "") if sig_col else "")
    has_signal = bool(sig.strip()) and sig.strip().lower() not in ("nan", "none")
    secreted = (
        "secreted" in kw
        or "kw-0964" in kw
        or "extracellular" in loc
        or "secreted" in loc
        or has_signal
    )
    return secreted, has_signal


def annotate_secreted(stable: pd.DataFrame, uniprot: pd.DataFrame) -> pd.DataFrame:
    out = stable.copy()
    out["secreted"] = False
    out["has_signal_peptide"] = False
    out["uniprot_keywords"] = ""
    out["uniprot_location"] = ""
    if uniprot.empty or ("error" in uniprot.columns and len(uniprot.columns) <= 2):
        return out
    acc_col = _col(uniprot, "entry") or _col(uniprot, "accession")
    if acc_col is None:
        return out
    u = uniprot.loc[uniprot[acc_col].astype(str).str.len() > 0].copy()
    u[acc_col] = u[acc_col].astype(str)
    kw_col = _col(u, "keyword")
    loc_col = _col(u, "subcellular", "location")
    sig_col = _col(u, "signal")
    flags = u.apply(lambda row: _is_secreted(row, kw_col, loc_col, sig_col), axis=1, result_type="expand")
    u["secreted"] = flags[0].astype(bool)
    u["has_signal_peptide"] = flags[1].astype(bool)
    lookup = u.drop_duplicates(acc_col).set_index(acc_col)
    for idx, acc in out["accession"].astype(str).items():
        if acc not in lookup.index:
            continue
        hit = lookup.loc[acc]
        out.at[idx, "secreted"] = bool(hit.get("secreted", False))
        out.at[idx, "has_signal_peptide"] = bool(hit.get("has_signal_peptide", False))
        if kw_col:
            out.at[idx, "uniprot_keywords"] = str(hit.get(kw_col, ""))
        if loc_col:
            out.at[idx, "uniprot_location"] = str(hit.get(loc_col, ""))
    return out


def convert_human_orthologs(accessions: list[str], cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    unique = sorted({str(a) for a in accessions if str(a) and str(a) != "nan"})
    if not unique:
        empty = pd.DataFrame()
        empty.to_csv(cache_path, index=False)
        return empty
    try:
        from gprofiler import GProfiler

        gp = GProfiler(return_dataframe=True)
        conv = gp.orth(organism="mmusculus", query=unique, target="hsapiens")
    except Exception as exc:  # noqa: BLE001
        conv = pd.DataFrame([{"error": str(exc)}])
    conv.to_csv(cache_path, index=False)
    json_path = cache_path.with_suffix(".json")
    json_path.write_text(conv.to_json(orient="records"), encoding="utf-8")
    return conv


def _extracellular_accessions(enrichment: pd.DataFrame) -> set[str]:
    if enrichment.empty or "source" not in enrichment.columns:
        return set()
    cc = enrichment.loc[enrichment["source"].astype(str) == "GO:CC"].copy()
    if cc.empty or "name" not in cc.columns:
        return set()
    hits = cc.loc[cc["name"].astype(str).str.lower().str.contains(EXTRACELLULAR_NAME, na=False)]
    accs: set[str] = set()
    inter_col = next((c for c in hits.columns if c.lower() in {"intersections", "intersection"}), None)
    if inter_col is None:
        return accs
    for raw in hits[inter_col].dropna().astype(str):
        for token in raw.replace(";", ",").split(","):
            tok = token.strip()
            if tok and tok.lower() not in {"nan", "none"}:
                accs.add(tok)
    return accs


def write_clock_annotation(
    membership: pd.DataFrame,
    de_dir: Path,
    stab_dir: Path,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    bio_dir = out_dir / "biology"
    cache_dir = bio_dir / "gprofiler"
    bio_dir.mkdir(parents=True, exist_ok=True)
    set_frames = []
    enrich_frames = []
    secreted_frames = []
    organs = [o for o in ORGAN_ORDER if o in membership["organ"].unique()]
    for organ in organs:
        print(f"  annotate {organ} ...", flush=True)
        organ_mem = membership.loc[membership["organ"] == organ]
        stability = _load_stability(stab_dir, organ)
        de_path = de_dir / f"{organ}_de.csv"
        de = pd.read_csv(de_path) if de_path.exists() else pd.DataFrame(columns=["accession"])
        if not de.empty:
            de["accession"] = de["accession"].astype(str)
            background = de["accession"].tolist()
        else:
            background = organ_mem["accession"].astype(str).tolist()
        for cutoff in (STABLE_CUTOFF, STABLE_STRICT):
            sets = protein_sets(organ_mem, stability, de, cutoff)
            labeled = sets["all"].copy()
            labeled.insert(0, "organ", organ)
            labeled.insert(1, "stable_cutoff", cutoff)
            set_frames.append(labeled)
            if cutoff != STABLE_CUTOFF:
                continue
            try:
                enrich_frames.append(enrich_sets(organ, sets, background, cache_dir))
            except Exception as exc:  # noqa: BLE001
                enrich_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "organ": organ,
                                "set": "all",
                                "n_query": 0,
                                "name": str(exc),
                                "note": "gprofiler_error",
                            }
                        ]
                    )
                )
            stable_rows = pd.concat([sets["stable_only"], sets["intersection"]], ignore_index=True)
            try:
                uni = fetch_uniprot_secreted(
                    stable_rows["accession"].tolist(),
                    bio_dir / "uniprot" / f"{organ}_stable.tsv.csv",
                )
            except Exception as exc:  # noqa: BLE001
                uni = pd.DataFrame([{"Entry": "", "error": str(exc)}])
            flagged = annotate_secreted(stable_rows, uni)
            flagged.insert(0, "organ", organ)
            secreted_frames.append(flagged)
            secreted_acc = flagged.loc[flagged["secreted"], "accession"].astype(str).tolist()
            convert_human_orthologs(secreted_acc, cache_dir / f"{organ}_secreted_human.csv")

    sets_out = pd.concat(set_frames, ignore_index=True)
    enrich_out = pd.concat(enrich_frames, ignore_index=True) if enrich_frames else pd.DataFrame()
    secreted_out = pd.concat(secreted_frames, ignore_index=True) if secreted_frames else pd.DataFrame()

    extra = _extracellular_accessions(enrich_out)
    if not secreted_out.empty:
        secreted_out["go_cc_extracellular"] = secreted_out["accession"].astype(str).isin(extra)
        human_frames = []
        for organ in organs:
            path = cache_dir / f"{organ}_secreted_human.csv"
            if not path.exists():
                continue
            conv = pd.read_csv(path)
            if conv.empty or "error" in conv.columns:
                continue
            incoming = _col(conv, "incoming") or conv.columns[0]
            name_col = _col(conv, "name")
            tmp = conv[[incoming]].copy()
            tmp["organ"] = organ
            tmp["human_ortholog"] = conv[name_col].astype(str) if name_col else ""
            tmp = tmp.rename(columns={incoming: "accession"})
            human_frames.append(tmp.drop_duplicates(["organ", "accession"]))
        if human_frames:
            human = pd.concat(human_frames, ignore_index=True)
            secreted_out = secreted_out.merge(human, on=["organ", "accession"], how="left")
        else:
            secreted_out["human_ortholog"] = ""
        secreted_out["circulating_reporter_candidate"] = (
            secreted_out["secreted"].astype(bool) | secreted_out["go_cc_extracellular"].astype(bool)
        )
        secreted_out["note"] = np.where(
            secreted_out["circulating_reporter_candidate"],
            "candidate circulating reporter (not measured plasma)",
            "",
        )

    summary = (
        sets_out.groupby(["organ", "stable_cutoff", "set"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    sign_summary = (
        sets_out.loc[sets_out["set"].isin(["stable_only", "intersection"])]
        .groupby(["organ", "stable_cutoff", "coef_sign"])
        .size()
        .rename("n")
        .reset_index()
    )
    sets_out.to_csv(bio_dir / "clock_protein_sets.csv", index=False)
    summary.to_csv(bio_dir / "clock_set_sizes.csv", index=False)
    sign_summary.to_csv(bio_dir / "stable_coef_signs.csv", index=False)
    enrich_out.to_csv(bio_dir / "gprofiler_enrichment.csv", index=False)
    secreted_out.to_csv(bio_dir / "stable_secreted.csv", index=False)
    n_sec = int(secreted_out["secreted"].sum()) if not secreted_out.empty and "secreted" in secreted_out.columns else 0
    print(f"  stable secreted candidates: {n_sec}", flush=True)
    return {
        "sets": sets_out,
        "set_sizes": summary,
        "signs": sign_summary,
        "enrichment": enrich_out,
        "secreted": secreted_out,
    }
