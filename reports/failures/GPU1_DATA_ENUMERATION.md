# GPU-1 — dataset enumeration stall

## Symptom / evidence

The pinned FineWeb-Edu `sample-10BT` loader retrieved its dataset card but emitted no
documents for several minutes. The zero-byte acquisition file was retained as
`data/raw/minillm-fineweb-edu.metadata-stall.jsonl`. No training used this file.

## Root cause

The stall was in dataset discovery, before any row was returned. The exact network
sub-request was not instrumented, so the root cause is not asserted more narrowly.

## Fix

Use the same upstream revision, with the explicit first sample-10BT Parquet shard
`sample/10BT/000_00000.parquet`; stream only its first 30,000 rows. Record that shard and
row selection in the manifest, rather than enumerate the whole dataset's glob patterns.

## Before / after

Before: zero rows delivered. After: 30,000 rows; 147,782,794 raw UTF-8 text bytes;
2 normalized exact duplicates removed. This is an ingestion fix, not a data-source change.
