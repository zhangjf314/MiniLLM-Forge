from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from minillm_forge.experiments_gpu4b_r1.diagnostics import (
    RELATIVE_FLOOR,
    gradient_metrics,
    tensor_metrics,
)
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.model.attention import repeat_key_value

REPO = Path(__file__).resolve().parents[1]
R1 = REPO / "artifacts/gpu4b_r1"


def _json(name: str) -> dict:
    return json.loads((R1 / name).read_text(encoding="utf-8"))


def test_protocol_freezes_gpu4b_start_and_checkpoint() -> None:
    protocol = _json("protocol.json")
    assert protocol["starting_commit"] == "805d6f44a114cb0cf826e474f5bc7354fd1bbd18"
    assert protocol["checkpoint"]["sha256"] == (
        "ee306384fcdef646f557cf9aa1165c464b2ca373c6d64d81f5df3b19b8f859fc"
    )


def test_protocol_freezes_original_input_and_tolerances() -> None:
    protocol = _json("protocol.json")
    assert protocol["fixed_input"]["batch_size"] == 2
    assert protocol["fixed_input"]["seed"] == 4204
    assert protocol["frozen_gpu4b_tolerances"]["bf16"] == {"atol": 0.08, "rtol": 0.03}


def test_tensor_metrics_elementwise_allclose_fraction() -> None:
    left = torch.tensor([1.0, 2.0, 3.0, 4.0])
    right = torch.tensor([1.0, 2.2, 3.0, 4.4])
    metric = tensor_metrics(left, right, atol=0.05, rtol=0.0)
    assert metric["allclose_fraction"] == 0.5
    assert not metric["allclose"]


def test_tensor_metrics_near_zero_policy_is_stable() -> None:
    left = torch.tensor([0.0, RELATIVE_FLOOR / 2, 2.0])
    right = torch.tensor([1.0, 1.0, 2.2])
    metric = tensor_metrics(left, right, atol=0.0, rtol=0.0)
    assert metric["near_zero_reference_count"] == 2
    assert metric["max_relative_error"] == pytest.approx(0.1)


def test_tensor_metrics_records_true_max_location() -> None:
    left = torch.zeros(2, 3)
    right = torch.tensor([[0.0, 0.0, 2.0], [0.0, -4.0, 0.0]])
    metric = tensor_metrics(left, right, atol=0.0, rtol=0.0)
    assert metric["max_error_index"] == [1, 1]
    assert metric["max_error_candidate"] == -4.0


def test_gradient_metrics_reports_cosine_and_rms() -> None:
    metric = gradient_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert metric["cosine_similarity"] == pytest.approx(1.0)
    assert metric["rms_error"] == 0.0


def test_gqa_8_4_mapping_is_group_preserving() -> None:
    value = torch.arange(4).view(1, 4, 1, 1)
    expanded = repeat_key_value(value, 2)
    assert expanded.flatten().tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_model_contract_keeps_rope_tied_embeddings_and_dimensions() -> None:
    config = MiniLLMConfig(
        vocab_size=24000,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1536,
        max_seq_len=1024,
    )
    model = MiniLLM(config)
    assert model.num_parameters() == 37_462_528
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.layers[0].self_attn.rope is not None


def test_causal_and_padding_mask_contract_remains_in_production_source() -> None:
    source = (REPO / "src/minillm_forge/model/attention.py").read_text(encoding="utf-8")
    assert "causal.view(1, 1, seq_len, seq_len) & key_padding_mask" in source
    assert "scores.masked_fill(~allowed" in source


def test_attention_scaling_contract_is_unchanged() -> None:
    source = (REPO / "src/minillm_forge/model/attention.py").read_text(encoding="utf-8")
    assert "self.scale = self.head_dim**-0.5" in source
    assert "scale=self.scale" in source


def test_result_reproduces_fp32_reference() -> None:
    result = _json("gpu4b_r1_result.json")
    assert result["original_gate_reproduction"]["fp32_pass"] is True


def test_result_reproduces_bf16_gate_failure() -> None:
    result = _json("gpu4b_r1_result.json")
    assert result["original_gate_reproduction"]["bf16_pass"] is False


def test_forward_records_qkv_rope_shapes_and_dtypes() -> None:
    tensors = _json("forward_diagnostics.json")["bf16"]["tensors"]
    assert tensors["block.0.q_projection"]["shape"] == [2, 65, 512]
    assert tensors["block.0.k_projection"]["shape"] == [2, 65, 256]
    assert tensors["block.0.rope.q"]["shape"] == [2, 8, 65, 64]
    assert tensors["block.0.rope.k"]["shape"] == [2, 4, 65, 64]
    assert tensors["block.0.rope.q"]["reference_dtype"] == "bfloat16"


def test_forward_localizes_first_difference_after_rope() -> None:
    result = _json("gpu4b_r1_result.json")
    assert result["bf16_forward_analysis"]["first_nonzero_public_boundary"] == (
        "block.0.attention.context"
    )


def test_precision_reconstruction_matches_manual() -> None:
    precision = _json("precision_experiment.json")
    metric = precision["variants"]["production_equivalent_manual"]["versus_actual_manual"]
    assert metric["max_abs_error"] == 0.0


def test_tied_embedding_remains_tied_and_decomposition_recomposes() -> None:
    tied = _json("gradient_update_diagnostics.json")["tied_embedding_decomposition"]
    assert tied["weight_is_tied_by_model_contract"] is True
    assert tied["manual_recomposition_max_abs"] < 1e-7
    assert tied["math_recomposition_max_abs"] < 1e-7


def test_adamw_single_step_was_recorded() -> None:
    result = _json("gradient_update_diagnostics.json")
    tied = result["optimizer_updates"]["token_embedding.weight"]
    assert tied["two_times_learning_rate"] == 0.0002
    assert tied["candidate_update_rms"] > 0
    assert set(result["optimizer_states"]["token_embedding.weight"]) == {"exp_avg", "exp_avg_sq"}


def test_production_attention_and_defaults_were_not_changed() -> None:
    result = _json("gpu4b_r1_result.json")
    assert result["production_attention_changed"] is False
    assert result["formal_training_default_changed"] is False
    config_source = (REPO / "src/minillm_forge/model/config.py").read_text(encoding="utf-8")
    assert 'attention_backend: str = "manual"' in config_source


def test_historical_artifact_hashes_are_preserved() -> None:
    protocol = _json("protocol.json")
    for relative, expected in protocol["historical_frozen_artifacts"].items():
        actual = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_diagnostics_do_not_execute_performance_or_compile() -> None:
    result = _json("gpu4b_r1_result.json")
    assert result["performance_benchmark_executed"] is False
    assert result["torch_compile_executed"] is False


def test_all_required_reports_exist() -> None:
    required = {
        "NUMERICAL_PATH_AUDIT.md", "FORWARD_ERROR_LOCALIZATION.md", "BF16_PRECISION_EXPERIMENT.md",
        "GRADIENT_AND_UPDATE_AUDIT.md", "ROOT_CAUSE_REPORT.md", "gpu4b_r1_result.json",
    }
    assert all((R1 / name).is_file() for name in required)
