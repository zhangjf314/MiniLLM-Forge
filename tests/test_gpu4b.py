from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from minillm_forge.experiments_gpu4b.audit import BACKENDS, TOLERANCES
from minillm_forge.experiments_gpu4b.benchmark import (
    BenchmarkConfig,
    prepare_cuda_measurement,
    run_warmups,
    summarize_measurements,
    write_gate_blocked_benchmarks,
)
from minillm_forge.model import GroupedQueryAttention, MiniLLM, MiniLLMConfig
from minillm_forge.model.attention import repeat_key_value

REPO = Path(__file__).resolve().parents[1]
GPU4B = REPO / "artifacts/gpu4b"


def _json(name: str) -> dict:
    return json.loads((GPU4B / name).read_text(encoding="utf-8"))


def _config(backend: str = "manual") -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        attention_backend=backend,
    )


def _attention_pair() -> tuple[GroupedQueryAttention, GroupedQueryAttention]:
    torch.manual_seed(42)
    manual = GroupedQueryAttention(_config("manual")).eval()
    math = GroupedQueryAttention(_config("sdpa_math")).eval()
    math.load_state_dict(manual.state_dict(), strict=True)
    return manual, math


def test_historical_attention_default_remains_manual() -> None:
    assert MiniLLMConfig().attention_backend == "manual"


def test_explicit_attention_backend_values_are_frozen() -> None:
    assert BACKENDS == ("manual", "sdpa_math", "sdpa_auto", "sdpa_flash")


def test_unknown_attention_backend_fails_cleanly() -> None:
    with pytest.raises(ValueError, match="unsupported attention_backend"):
        _config("renamed-fallback")


def test_gqa_expansion_preserves_group_order() -> None:
    values = torch.tensor([[[[1.0]], [[2.0]]]])
    assert repeat_key_value(values, 2).flatten().tolist() == [1.0, 1.0, 2.0, 2.0]


def test_manual_and_sdpa_math_fp32_forward() -> None:
    manual, math = _attention_pair()
    inputs = torch.randn(2, 9, 32)
    mask = torch.tensor([[1] * 9, [1] * 7 + [0] * 2], dtype=torch.bool)
    torch.testing.assert_close(manual(inputs, mask), math(inputs, mask), **TOLERANCES["fp32"])


def test_bf16_forward_uses_frozen_tolerance_record() -> None:
    result = _json("attention_correctness.json")
    assert result["tolerances"]["bf16"] == TOLERANCES["bf16"]
    assert "max_abs_error" in result["bf16"]["sdpa_math"]["logits"]


def test_gqa_native_math_and_expanded_auto_shapes_match() -> None:
    inputs = torch.randn(2, 7, 32)
    outputs = [GroupedQueryAttention(_config(name))(inputs) for name in ("sdpa_math", "sdpa_auto")]
    assert outputs[0].shape == outputs[1].shape == (2, 7, 32)


def test_rope_preserves_q_and_k_dimensions() -> None:
    module = GroupedQueryAttention(_config())
    query = module._shape(module.q_proj(torch.randn(2, 7, 32)), module.num_heads)
    key = module._shape(module.k_proj(torch.randn(2, 7, 32)), module.num_kv_heads)
    positions = torch.arange(7).expand(2, -1)
    rotated_q, rotated_k = module.rope(query, key, positions)
    assert rotated_q.shape == query.shape
    assert rotated_k.shape == key.shape


def test_causal_mask_blocks_future_token_for_sdpa_math() -> None:
    torch.manual_seed(7)
    model = MiniLLM(_config("sdpa_math")).eval()
    first = torch.tensor([[1, 8, 9, 10, 11]])
    changed = torch.tensor([[1, 8, 9, 10, 27]])
    with torch.no_grad():
        left = model(first).logits
        right = model(changed).logits
    torch.testing.assert_close(left[:, :4], right[:, :4], atol=1e-6, rtol=1e-6)


def test_attention_scale_remains_inverse_sqrt_head_dim() -> None:
    module = GroupedQueryAttention(_config())
    assert module.scale == pytest.approx(module.head_dim**-0.5)


def test_sdpa_math_backward_is_finite() -> None:
    module = GroupedQueryAttention(_config("sdpa_math"))
    module(torch.randn(2, 7, 32)).sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_sdpa_auto_backward_is_finite() -> None:
    module = GroupedQueryAttention(_config("sdpa_auto"))
    module(torch.randn(2, 7, 32)).sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_flash_backend_was_actually_qualified() -> None:
    trial = _json("environment_audit.json")["backend_qualification"]["sdpa_flash_expanded_gqa"]
    assert trial["qualified"] is False
    assert "No available kernel" in trial["error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unavailable_flash_fails_without_fallback() -> None:
    module = GroupedQueryAttention(_config("sdpa_flash")).cuda().to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="No available kernel"):
        module(torch.randn(1, 7, 32, device="cuda", dtype=torch.bfloat16))


def test_flash_result_is_not_mislabeled_as_success() -> None:
    result = _json("attention_correctness.json")
    assert result["backend_gates"]["sdpa_flash"] == "FLASH_BACKEND_NOT_AVAILABLE"


def test_checkpoint_parameter_schema_is_backend_independent() -> None:
    keys = [list(MiniLLM(_config(backend)).state_dict()) for backend in BACKENDS]
    assert all(item == keys[0] for item in keys[1:])


def test_manual_state_loads_strictly_into_sdpa() -> None:
    manual = MiniLLM(_config("manual"))
    candidate = MiniLLM(_config("sdpa_auto"))
    candidate.load_state_dict(manual.state_dict(), strict=True)


def test_greedy_generation_regression_is_exact() -> None:
    comparisons = _json("attention_correctness.json")["generation"]["comparisons"]
    assert comparisons["sdpa_math"]["token_ids_identical"] is True
    assert comparisons["sdpa_auto"]["token_ids_identical"] is True


def test_eos_regression_is_exact() -> None:
    comparisons = _json("attention_correctness.json")["generation"]["comparisons"]
    assert comparisons["sdpa_math"]["eos_identical"] is True
    assert comparisons["sdpa_auto"]["eos_identical"] is True


def test_fp32_reference_gate_passed() -> None:
    assert _json("attention_correctness.json")["fp32"]["sdpa_math"]["pass"] is True


def test_short_training_trajectory_is_finite_and_aligned() -> None:
    trajectories = _json("attention_correctness.json")["short_trajectory"]
    for backend in ("sdpa_math", "sdpa_auto"):
        assert trajectories[backend]["pass"] is True


def test_sdpa_checkpoint_loads_eager_manual() -> None:
    assert _json("attention_correctness.json")["checkpoint"]["pass"] is True


def test_compile_required_modes_have_explicit_results() -> None:
    modes = _json("compile_qualification.json")["modes"]
    assert set(modes) == {"default", "reduce-overhead", "max-autotune"}


def test_compile_default_failure_is_clean() -> None:
    default = _json("compile_qualification.json")["modes"]["default"]
    assert default["status"] == "NOT_SUPPORTED"
    assert default["error_type"] == "TritonMissing"


def test_compile_reduce_overhead_failure_is_clean() -> None:
    reduced = _json("compile_qualification.json")["modes"]["reduce-overhead"]
    assert reduced["status"] == "NOT_SUPPORTED"
    assert reduced["error_type"] == "TritonMissing"


def test_compile_max_autotune_obeys_prerequisite() -> None:
    maximum = _json("compile_qualification.json")["modes"]["max-autotune"]
    assert maximum["status"] == "SKIPPED_PREREQUISITE"


def test_compile_failure_does_not_claim_eager_fallback() -> None:
    modes = _json("compile_qualification.json")["modes"]
    assert all(item["silent_eager_fallback"] is False for item in modes.values())


def test_compile_correctness_is_not_claimed_without_compilation() -> None:
    result = _json("compile_qualification.json")
    assert result["qualified_modes"] == []
    assert result["correctness_status"] == "NOT_RUN_NO_QUALIFIED_MODE"


def test_benchmark_configuration_hash_is_stable() -> None:
    config = BenchmarkConfig("manual", 1, 256, "bfloat16", 5, 20, "forward")
    assert config.sha256 == config.sha256
    assert len(config.sha256) == 64


def test_sequence_length_changes_benchmark_hash() -> None:
    left = BenchmarkConfig("manual", 1, 256, "bfloat16", 5, 20, "forward")
    right = BenchmarkConfig("manual", 1, 512, "bfloat16", 5, 20, "forward")
    assert left.sha256 != right.sha256


def test_warmup_count_is_not_returned_as_measurements() -> None:
    calls = []
    run_warmups(lambda: calls.append("warmup"), 3, lambda: calls.append("sync"))
    assert calls == ["warmup", "warmup", "warmup", "sync"]


def test_cuda_measurement_preparation_syncs_then_resets_peak() -> None:
    calls = []
    prepare_cuda_measurement(lambda: calls.append("sync"), lambda: calls.append("reset"))
    assert calls == ["sync", "reset"]


def test_measurement_summary_reports_distribution() -> None:
    summary = summarize_measurements([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["median"] == 3.0
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 5.0
    assert summary["cv"] > 0


def test_blocked_benchmark_matrix_keeps_lengths_independent() -> None:
    result = write_gate_blocked_benchmarks(REPO)
    records = result["attention_only"]
    assert len(records) == 16
    assert {row["sequence_length"] for row in records} == {256, 512, 768, 1024}
    assert all(row["status"] == "NOT_RUN" for row in records)


def test_blocked_benchmarks_retain_backend_identity() -> None:
    records = write_gate_blocked_benchmarks(REPO)["attention_only"]
    assert {row["backend"] for row in records} == set(BACKENDS)
    assert all(len(row["config_sha256"]) == 64 for row in records)


def test_benchmark_status_vocabulary_includes_oom_and_timeout() -> None:
    source = (REPO / "src/minillm_forge/experiments_gpu4b/benchmark.py").read_text()
    assert '"status": "OOM"' in source
    assert '"status": "TIMEOUT"' in source


def test_auto_backend_dispatch_was_observed_not_assumed() -> None:
    auto = _json("environment_audit.json")["backend_qualification"]["sdpa_auto_production_path"]
    assert auto["selected"] == "efficient_attention"
    assert any("efficient_attention" in operator for operator in auto["operators"])


def test_baseline_manifest_rechecks_formal_checkpoint() -> None:
    manifest = _json("baseline_manifest.json")
    checkpoint = manifest["formal_pretrained_checkpoint"]
    assert (
        checkpoint["sha256"]
        == hashlib.sha256((REPO / "runs/E01-minillm-formal/best.pt").read_bytes()).hexdigest()
    )


def test_gpu4a_frozen_artifacts_remain_unchanged() -> None:
    protocol = _json("protocol.json")
    for relative, expected in protocol["historical_frozen_inputs"].items():
        digest = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        assert digest == expected


def test_required_audit_and_correctness_reports_exist() -> None:
    for name in (
        "ENVIRONMENT_AND_BACKEND_AUDIT.md",
        "ATTENTION_IMPLEMENTATION_AUDIT.md",
        "ATTENTION_CORRECTNESS.md",
        "COMPILE_CORRECTNESS.md",
        "COMPILE_BENCHMARK.md",
    ):
        assert (GPU4B / name).is_file()
