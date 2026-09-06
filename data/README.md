# Data layout

Training corpora are not committed. Keep immutable downloads under `data/raw/`, filtered
and deduplicated files under `data/processed/`, and never place `data/eval/` records in a
training path.

Expected local inputs:

- `data/processed/fineweb_edu_train.txt`: one normalized document per line
- `data/processed/openr1_math_sft.jsonl`: objects with `problem` and `solution`
- `data/eval/controlled_math.jsonl`: repository-owned held-out evaluation only

Every processed file needs a matching manifest under `artifacts/data_manifests/` with a
pinned upstream revision, config, sample/token counts, SHA-256 digest, and overlap count.

