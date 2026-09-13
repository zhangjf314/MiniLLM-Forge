import torch

from minillm_forge.execution.gpu2c import _digest_value, _snapshot_comparison


def test_gpu2c_state_digest_is_content_sensitive_and_dict_order_invariant():
    left = {"b": torch.tensor([1, 2]), "a": [3, 4]}
    right = {"a": [3, 4], "b": torch.tensor([1, 2])}
    changed = {"a": [3, 4], "b": torch.tensor([1, 3])}
    assert _digest_value(left) == _digest_value(right)
    assert _digest_value(left) != _digest_value(changed)


def test_gpu2c_snapshot_comparison_reports_each_field():
    result = _snapshot_comparison({"a": 1, "b": 2}, {"a": 1, "b": 3})
    assert result == {"all_match": False, "field_matches": {"a": True, "b": False}}
