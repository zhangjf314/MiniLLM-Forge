from minillm_forge.config import config_hash, load_config


def test_config_inheritance_deep_merges_nested_sections(tmp_path):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("model:\n  hidden: 32\n  layers: 2\ntraining:\n  seed: 7\n", encoding="utf-8")
    child.write_text("extends: parent.yaml\nmodel:\n  layers: 4\n", encoding="utf-8")
    loaded = load_config(child)
    assert loaded == {"model": {"hidden": 32, "layers": 4}, "training": {"seed": 7}}
    assert config_hash(loaded) == config_hash(load_config(child))
