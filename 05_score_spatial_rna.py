#!/usr/bin/env python3
"""Visium section clocks, niche scores, and hotspots."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paths import project_root
from src.spatial_clocks import write_clocks
from src.spatial_epicenters import main as hotspot_main
from src.spatial_followup import spatial_pass
from src.spatial_frozen import write_frozen_sections, write_niches
from src.spatial_io import write_pseudobulk


def main() -> None:
    root = project_root()
    gene_map = root / "results" / "human_clock_overlap" / "mouse_stable_with_genes.csv"
    if not gene_map.exists():
        raise FileNotFoundError(
            f"Missing {gene_map}. It maps core clock proteins to gene symbols."
        )
    data_dir = root / "Dataset" / "Spatial_Transcriptome" / "GSE283283_aging"
    out_dir = root / "results" / "spatial_aging"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Section profiles ...", flush=True)
    write_pseudobulk(data_dir, out_dir)
    print("Spatial RNA clocks ...", flush=True)
    write_clocks(out_dir, root / "results")
    print("Fixed proteomic weights on sections ...", flush=True)
    write_frozen_sections(out_dir, root / "results")
    print("Niches ...", flush=True)
    write_niches(out_dir, root / "results")
    print("White-matter enrichment ...", flush=True)
    spatial_pass(pd.read_csv(gene_map))
    print("Hotspots ...", flush=True)
    hotspot_main()
    print(f"Wrote spatial results under {out_dir}")


if __name__ == "__main__":
    main()
