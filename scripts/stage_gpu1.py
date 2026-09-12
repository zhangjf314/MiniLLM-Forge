"""Workspace-local cache bootstrap; run with .venv/Scripts/python.exe."""

import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
for key, value in {
    "HF_HOME": ".hf-cache",
    "HF_HUB_CACHE": ".hf-cache/hub",
    "HF_DATASETS_CACHE": ".hf-cache/datasets",
    "UV_CACHE_DIR": ".uv-cache",
    "MPLCONFIGDIR": ".mpl-cache",
    "TEMP": ".tmp",
    "TMP": ".tmp",
}.items():
    directory = root / value
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["prepare"]:
        from minillm_forge.data.formal import prepare

        prepare()
    else:
        from minillm_forge.cli.formal_pretrain import main

        main()
