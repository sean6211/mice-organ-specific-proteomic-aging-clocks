"""Load GSE283283 Visium sections and collapse each section to one mouse."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

FILENAME_RE = re.compile(
    r"(GSM\d+)_AGE_(\d+)_(\d+)_([A-Za-z0-9]+)_((?:GEX|SS)\w+)_([A-D]1)_"
)
# 52.1775 weeks per year / 12 months. 17, 73, and 112 weeks are 3.9, 16.8, and 25.8 months.
WEEKS_PER_MONTH = 52.1775 / 12.0
SECTION_ORGANS = [
    "Brain1",
    "Brain2",
    "Brain3",
    "Brain4",
    "Brain5",
    "Heart",
    "Kidney",
    "Liver",
    "Lung",
    "Spleen",
]


def age_months(age_weeks: float) -> float:
    return float(age_weeks) / WEEKS_PER_MONTH


def sample_table(root: Path) -> pd.DataFrame:
    """One row per Visium section, parsed from the Space Ranger file name."""
    rows = []
    for h5_path in sorted(root.glob("*/*filtered_feature_bc_matrix.h5")):
        match = FILENAME_RE.search(h5_path.name)
        if match is None:
            raise ValueError(f"Unrecognized Visium file name: {h5_path.name}")
        gsm, mouse, weeks, organ, slide, area = match.groups()
        positions = h5_path.with_name(h5_path.name.replace("filtered_feature_bc_matrix.h5", "tissue_positions.csv.gz"))
        if not positions.exists():
            raise FileNotFoundError(positions)
        rows.append(
            {
                "gsm": gsm,
                "mouse_id": int(mouse),
                "age_weeks": int(weeks),
                "age_months": age_months(int(weeks)),
                "organ": organ,
                "slide": slide,
                "area": area,
                "h5_path": str(h5_path),
                "positions_path": str(positions),
            }
        )
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise FileNotFoundError(f"No Visium matrices under {root}")
    duplicated = samples.duplicated(["organ", "mouse_id"], keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate organ-mouse sections:\n{samples.loc[duplicated]}")
    return samples.sort_values(["organ", "age_weeks", "mouse_id"]).reset_index(drop=True)


def _collapse_genes(matrix: sparse.spmatrix, names: np.ndarray) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Sum rows that share a gene symbol. Returns genes by spots."""
    names = np.asarray(names, dtype=str)
    unique, inverse = np.unique(names, return_inverse=True)
    if len(unique) == len(names):
        collapsed = matrix.tocsc()
    else:
        group = sparse.csr_matrix(
            (np.ones(len(names), dtype=np.float64), (inverse, np.arange(len(names)))),
            shape=(len(unique), len(names)),
        )
        collapsed = (group @ matrix).tocsc()
    order = np.argsort(unique)
    return collapsed[order], unique[order]


def load_section(h5_path: Path, positions_path: Path) -> dict:
    """Gene-by-spot counts for spots under tissue, with pixel coordinates."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        shape = tuple(int(value) for value in handle["matrix/shape"][:])
        matrix = sparse.csc_matrix(
            (
                handle["matrix/data"][:].astype(np.float64),
                handle["matrix/indices"][:],
                handle["matrix/indptr"][:],
            ),
            shape=shape,
        )
        names = np.array([value.decode() if isinstance(value, bytes) else str(value) for value in handle["matrix/features/name"][:]])
        barcodes = np.array(
            [value.decode() if isinstance(value, bytes) else str(value) for value in handle["matrix/barcodes"][:]]
        )
    matrix, names = _collapse_genes(matrix, names)
    positions = pd.read_csv(positions_path)
    positions["barcode"] = positions["barcode"].astype(str)
    coordinates = positions.set_index("barcode").reindex(barcodes)
    if coordinates[["pxl_col_in_fullres", "pxl_row_in_fullres"]].isna().any().any():
        missing = int(coordinates["pxl_col_in_fullres"].isna().sum())
        raise ValueError(f"{missing} barcodes in {h5_path.name} have no tissue position")
    return {
        "genes": names,
        "barcodes": barcodes,
        "counts": matrix,
        "pxl_col": coordinates["pxl_col_in_fullres"].to_numpy(dtype=float),
        "pxl_row": coordinates["pxl_row_in_fullres"].to_numpy(dtype=float),
    }


def section_totals(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Summed counts per gene symbol for one section."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        data = handle["matrix/data"][:].astype(np.float64)
        indices = handle["matrix/indices"][:]
        n_genes = int(handle["matrix/shape"][0])
        names = np.array([value.decode() if isinstance(value, bytes) else str(value) for value in handle["matrix/features/name"][:]])
    totals = np.bincount(indices, weights=data, minlength=n_genes)
    unique, inverse = np.unique(names, return_inverse=True)
    if len(unique) != len(names):
        totals = np.bincount(inverse, weights=totals)
    order = np.argsort(unique)
    return unique[order], totals[order]


def write_pseudobulk(root: Path, out_dir: Path) -> pd.DataFrame:
    """Write raw gene-by-mouse counts for each organ and the section table."""
    samples = sample_table(root)
    dest = out_dir / "pseudobulk"
    dest.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out_dir / "samples.csv", index=False)
    for organ, group in samples.groupby("organ", sort=True):
        columns = {}
        genes = None
        for row in group.sort_values("mouse_id").itertuples(index=False):
            symbols, totals = section_totals(Path(row.h5_path))
            if genes is None:
                genes = symbols
            elif not np.array_equal(genes, symbols):
                aligned = pd.Series(totals, index=symbols).reindex(genes).fillna(0.0)
                totals = aligned.to_numpy(dtype=float)
            columns[str(row.mouse_id)] = totals
        frame = pd.DataFrame(columns, index=genes)
        frame.index.name = "gene"
        frame.to_csv(dest / f"{organ}.csv")
        print(f"  pseudobulk {organ}: {frame.shape[0]} genes x {frame.shape[1]} mice", flush=True)
    return samples
