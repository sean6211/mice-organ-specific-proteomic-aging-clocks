#!/usr/bin/env python3
"""Age-differential overlap, core-protein overlap, and cross-organ transfer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ageaccel import ORGAN_ORDER
from src.clock_explain import cross_organ_overlap, organ_features
from src.de_test import write_de_overlap
from src.extract_mztab import ORGAN_FILES
from src.paths import project_root
from src.shared_program import shared_accessions, transfer_clocks


def load_saved_matrices(out_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    matrix_dir = out_dir / "matrices"
    samples_path = out_dir / "metadata" / "samples.csv"
    missing = [organ for organ in ORGAN_FILES if not (matrix_dir / f"{organ}_lfq.csv").exists()]
    if missing or not samples_path.exists():
        raise FileNotFoundError(
            "Saved matrices were not found. Run 01_train_discovery_clocks.py first."
        )
    matrices = {}
    for organ in ORGAN_FILES:
        frame = pd.read_csv(matrix_dir / f"{organ}_lfq.csv", index_col=0)
        rename = {column: str(column) for column in frame.columns if str(column) != "description"}
        matrices[organ] = frame.rename(columns=rename)
    samples = pd.read_csv(samples_path)
    samples["mouse_id"] = samples["mouse_id"].astype(str)
    return matrices, samples


def main() -> None:
    root = project_root()
    out_dir = root / "results"
    matrices, samples = load_saved_matrices(out_dir)
    coef_dir = out_dir / "clocks" / "coefficients"
    stab_dir = out_dir / "clocks" / "stability"

    print("Age-differential proteins ...", flush=True)
    _summary, membership = write_de_overlap(matrices, samples, coef_dir, stab_dir, out_dir)

    print("Core-protein contributions and overlap ...", flush=True)
    features = [
        organ_features(out_dir, organ, samples, membership)
        for organ in ORGAN_ORDER
        if organ in matrices
    ]
    features = pd.concat(features, ignore_index=True)
    pairs, multi = cross_organ_overlap(features)
    explain = out_dir / "clock_explain"
    explain.mkdir(parents=True, exist_ok=True)
    features.to_csv(explain / "stable_features.csv", index=False)
    pairs.to_csv(explain / "cross_organ_overlap.csv", index=False)
    multi.to_csv(explain / "proteins_in_several_organs.csv", index=False)

    print("Cross-organ transfer on proteins detected in every organ ...", flush=True)
    shared = shared_accessions(matrices, samples)
    shared_dir = out_dir / "shared_program"
    shared_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(shared, name="accession").to_csv(shared_dir / "shared_accessions.csv", index=False)
    print(f"  {len(shared)} shared proteins", flush=True)
    transfer = transfer_clocks(matrices, samples, shared)
    transfer.to_csv(shared_dir / "transfer_clocks.csv", index=False)
    print(f"Wrote protein comparisons under {out_dir}")


if __name__ == "__main__":
    main()
