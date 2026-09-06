import json

import pytest

from minillm_forge.data.manifest import file_digest, validate_file_manifest


def test_file_manifest_detects_data_changes(tmp_path):
    data = tmp_path / "data.txt"
    manifest = tmp_path / "manifest.json"
    data.write_text("stable data\n", encoding="utf-8", newline="\n")
    manifest.write_text(json.dumps({"sha256": file_digest(data)}), encoding="utf-8")
    assert validate_file_manifest(manifest, data)["sha256"] == file_digest(data)
    data.write_text("changed data\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_file_manifest(manifest, data)
