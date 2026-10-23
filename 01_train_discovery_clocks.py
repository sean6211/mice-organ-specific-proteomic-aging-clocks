#!/usr/bin/env python3
"""Train the eight discovery clocks on the Scifo LFQ matrices."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extract_mztab import extract_all
from src.paths import project_root
from src.train_clocks import train_all


def main() -> None:
    root = project_root()
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices, samples = extract_all(out_dir)
    metrics, _predictions = train_all(matrices, samples, out_dir)
    print(metrics.to_string(index=False))
    print(f"Wrote clocks under {out_dir / 'clocks'}")


if __name__ == "__main__":
    main()
