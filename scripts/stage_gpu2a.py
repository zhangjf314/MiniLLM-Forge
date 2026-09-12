"""Workspace-local Stage GPU-2A entry point."""

import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
for key, value in {
    "HF_HOME": ".hf-cache",
    "HF_HUB_CACHE": ".hf-cache/hub",
    "HF_DATASETS_CACHE": ".hf-cache/datasets",
    "MPLCONFIGDIR": ".mpl-cache",
}.items():
    os.environ.setdefault(key, value)

from minillm_forge.cli.stage_gpu2a import main  # noqa: E402

if __name__ == "__main__":
    main()
