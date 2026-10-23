"""Locate the directory that contains Dataset/."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Directory that contains Dataset/. Override with PROTEOMIC_CLOCK_ROOT."""
    env = os.environ.get("PROTEOMIC_CLOCK_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "Dataset").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find Dataset/. Set PROTEOMIC_CLOCK_ROOT to the project root."
    )
