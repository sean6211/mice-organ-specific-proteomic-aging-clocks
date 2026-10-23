#!/usr/bin/env python3
"""Score Takasugi and Shavlakadze RNA with the fixed proteomic weights."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.gse314817_cross_organ import write_gse314817_cross_organ
from src.gse314817_transfer import write_gse314817_transfer
from src.paths import project_root
from src.rna_cross_organ import write_rna_cross_organ
from src.rna_portability import write_rna_portability
from src.takasugi_rna import TRANSFER_ORGANS, write_takasugi_rna


def main() -> None:
    root = project_root()
    out_dir = root / "results"
    gene_map = out_dir / "human_clock_overlap" / "mouse_stable_with_genes.csv"
    if not gene_map.exists():
        raise FileNotFoundError(
            f"Missing {gene_map}. It lists organ, accession, gene, "
            "selection_frequency, mean_coef and description for the core clock proteins."
        )
    missing = [
        organ
        for organ in TRANSFER_ORGANS
        if not (out_dir / "clocks" / "stability" / f"{organ}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing discovery clocks for {missing}. Run 01_train_discovery_clocks.py first."
        )

    takasugi = root / "Dataset" / "Takasugi"
    print("Takasugi paired RNA ...", flush=True)
    write_takasugi_rna(takasugi, out_dir)
    print("RNA null gene sets and weight controls ...", flush=True)
    write_rna_portability(takasugi, out_dir)
    print("Cross-organ Takasugi RNA ...", flush=True)
    write_rna_cross_organ(takasugi, out_dir)

    print("Shavlakadze RNA ...", flush=True)
    write_gse314817_transfer(root, out_dir)
    write_gse314817_cross_organ(root, out_dir)
    print(f"Wrote RNA transfers under {out_dir}")


if __name__ == "__main__":
    main()
