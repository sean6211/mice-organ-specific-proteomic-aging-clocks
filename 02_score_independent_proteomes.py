#!/usr/bin/env python3
"""Score Takasugi, Lu, Burns, and Watanabe proteomes with the fixed weights."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.burns_transfer import write_burns_interventions, write_burns_transfer
from src.lu_transfer import MATRIX_NAME, META_NAME, write_lu_transfer
from src.orwell_transfer import write_orwell_transfer
from src.paths import project_root
from src.takasugi_transfer import TRANSFER_ORGANS, write_takasugi_transfer


def _require_clocks(out_dir: Path) -> None:
    missing = [
        organ
        for organ in TRANSFER_ORGANS
        if not (out_dir / "clocks" / "stability" / f"{organ}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing discovery clocks for {missing}. Run 01_train_discovery_clocks.py first."
        )


def main() -> None:
    root = project_root()
    out_dir = root / "results"
    _require_clocks(out_dir)

    takasugi = root / "Dataset" / "Takasugi"
    print("Takasugi TMT ...", flush=True)
    write_takasugi_transfer(takasugi, out_dir)

    lu = root / "Dataset" / "Lu_intervention"
    if not (lu / MATRIX_NAME).exists() or not (lu / META_NAME).exists():
        raise FileNotFoundError(lu)
    print("Lu PulseDIA ...", flush=True)
    write_lu_transfer(lu, out_dir)

    burns = root / "Dataset" / "Burns"
    print("Burns interventions ...", flush=True)
    write_burns_transfer(burns, out_dir)
    write_burns_interventions(burns, out_dir)

    watanabe = root / "Dataset" / "Orwell_intervention"
    print("Watanabe liver ...", flush=True)
    write_orwell_transfer(watanabe, out_dir)
    print(f"Wrote proteome transfers under {out_dir}")


if __name__ == "__main__":
    main()
