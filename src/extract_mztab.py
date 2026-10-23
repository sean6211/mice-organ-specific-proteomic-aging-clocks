"""Stream mzTab MTD + PRT sections into per-organ LFQ matrices."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .paths import project_root

ROOT = project_root()
SCIFO_DIR = ROOT / "Dataset" / "Scifo"

ORGAN_FILES = {
    "brain": "Mar032021_AgeingMouse_Brain_LFQ_120min_60K_TopS.mzTab",
    "heart": "May072021_AgeingMouse_Heart_LFQ_120min_60K_TopS.mzTab",
    "lung": "Mar272021_AgeingMouseLungs_LFQ_120min_60K_TopS.mzTab",
    "liver": "April062021_AgeingMouse_Liver_LFQ_120min_60K_TopS.mzTab",
    "kidney": "April162021_AgeingMouse_Kidneys_LFQ_120min_60K_TopS.mzTab",
    "spleen": "Mar212021_AgeingMouse_Spleen_LFQ_120min_60K_TopS.mzTab",
    "muscle": "Juni232021_AgeingMouse_SkeletalMuscle_LFQ_120min_60K_TopS.mzTab",
    "testis": "Mai242021_AgeingMouse_Testis_LFQ_120min_60K_TopS.mzTab",
}

MS_RUN_RE = re.compile(r"^ms_run\[(\d+)\]-location$")
ASSAY_REF_RE = re.compile(r"^assay\[(\d+)\]-ms_run_ref$")
MS_RUN_IDX_RE = re.compile(r"ms_run\[(\d+)\]")
ASSAY_COL_RE = re.compile(r"^protein_abundance_assay\[(\d+)\]$")
SAMPLE_RE = re.compile(r"_(\d+[a-z]?)_(\d+)M_c(\d+)_", re.IGNORECASE)

NON_MOUSE_SPECIES = (
    "os=sus scrofa",
    "os=bos taurus",
    "os=homo sapiens",
    "os=gallus gallus",
    "os=equus caballus",
    "os=oryctolagus",
    "os=saccharomyces",
)


def parse_sample_name(location: str) -> tuple[int, int, str, str]:
    name = Path(str(location).replace("\\", "/")).stem
    match = SAMPLE_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse mouse/age/cage from {location!r}")
    mouse_token, age, cage = match.group(1), int(match.group(2)), match.group(3)
    mouse_id = int(re.match(r"\d+", mouse_token).group(0))
    return mouse_id, age, f"c{cage}", name


def _is_contaminant(accession: str, description: str) -> bool:
    acc = (accession or "").upper()
    desc = (description or "").lower()
    if acc.startswith(("CONTAMINANT", "CON_", "CRAP")):
        return True
    if any(tag in acc for tag in ("K1C", "K2C", "KRT")) and "os=mus musculus" not in desc:
        return True
    if any(species in desc for species in NON_MOUSE_SPECIES):
        return True
    return False


def _parse_abundance(value: str) -> float:
    if value in {"", "null", "NA", "NaN", "nan", "0", "0.0"}:
        return float("nan")
    try:
        number = float(value)
    except ValueError:
        return float("nan")
    if number == 0:
        return float("nan")
    return number


def extract_organ(organ: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms_run_locations: dict[int, str] = {}
    assay_to_run: dict[int, int] = {}
    header: list[str] | None = None
    assay_cols: dict[int, int] = {}
    rows: list[dict[str, object]] = []

    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("PEP\t") or line.startswith("PEH\t") or line.startswith("PSM\t"):
                break
            if line.startswith("MTD\t"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                _, key, value = parts[0], parts[1], parts[2]
                run_match = MS_RUN_RE.match(key)
                if run_match:
                    ms_run_locations[int(run_match.group(1))] = value
                    continue
                assay_match = ASSAY_REF_RE.match(key)
                if assay_match:
                    run_idx = MS_RUN_IDX_RE.search(value)
                    if run_idx:
                        assay_to_run[int(assay_match.group(1))] = int(run_idx.group(1))
                continue
            if line.startswith("PRH\t"):
                header = line.rstrip("\n").split("\t")
                assay_cols = {
                    int(match.group(1)): idx
                    for idx, col in enumerate(header)
                    if (match := ASSAY_COL_RE.match(col))
                }
                continue
            if not line.startswith("PRT\t") or header is None:
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < len(header):
                cols.extend([""] * (len(header) - len(cols)))
            accession = cols[header.index("accession")]
            description = cols[header.index("description")]
            if not accession or _is_contaminant(accession, description):
                continue
            record: dict[str, object] = {
                "accession": accession,
                "description": description,
            }
            for assay_idx, col_idx in assay_cols.items():
                record[f"assay_{assay_idx}"] = _parse_abundance(cols[col_idx])
            rows.append(record)

    if not rows:
        raise RuntimeError(f"No protein rows parsed from {path}")

    proteins = pd.DataFrame(rows).drop_duplicates(subset="accession", keep="first")
    sample_rows = []
    mouse_columns: dict[int, str] = {}
    for assay_idx in sorted(assay_cols):
        run_idx = assay_to_run.get(assay_idx, assay_idx)
        location = ms_run_locations[run_idx]
        mouse_id, age, cage, raw_stem = parse_sample_name(location)
        mouse_columns[assay_idx] = str(mouse_id)
        sample_rows.append(
            {
                "organ": organ,
                "mouse_id": mouse_id,
                "age_months": age,
                "cage": cage,
                "assay_index": assay_idx,
                "ms_run_index": run_idx,
                "raw_file": raw_stem,
            }
        )

    samples = pd.DataFrame(sample_rows).sort_values(["age_months", "mouse_id"])
    rename = {f"assay_{assay_idx}": mouse_id for assay_idx, mouse_id in mouse_columns.items()}
    matrix = proteins.rename(columns=rename).set_index("accession")
    descriptions = matrix.pop("description")
    ordered_mice = [str(m) for m in samples["mouse_id"].tolist()]
    matrix = matrix.reindex(columns=ordered_mice)
    matrix.insert(0, "description", descriptions)
    return matrix, samples


def extract_all(out_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    out_dir = out_dir or (ROOT / "results")
    matrix_dir = out_dir / "matrices"
    meta_dir = out_dir / "metadata"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    matrices: dict[str, pd.DataFrame] = {}
    sample_frames: list[pd.DataFrame] = []
    for organ, filename in ORGAN_FILES.items():
        path = SCIFO_DIR / filename
        print(f"Extracting {organ} from {path.name} ...", flush=True)
        matrix, samples = extract_organ(organ, path)
        matrices[organ] = matrix
        sample_frames.append(samples)
        matrix.to_csv(matrix_dir / f"{organ}_lfq.csv")
        print(
            f"  {organ}: {matrix.shape[0]} proteins, {samples.shape[0]} samples",
            flush=True,
        )

    samples = pd.concat(sample_frames, ignore_index=True)
    samples.to_csv(meta_dir / "samples.csv", index=False)
    mice = (
        samples.drop_duplicates("mouse_id")
        .sort_values(["age_months", "mouse_id"])
        [["mouse_id", "age_months", "cage"]]
    )
    mice.to_csv(meta_dir / "mice.csv", index=False)
    return matrices, samples
