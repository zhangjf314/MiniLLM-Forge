# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

from minillm_forge.experiments_gpu4a.runtime import file_sha256, read_json, write_json
from minillm_forge.experiments_gpu4b.benchmark import write_gate_blocked_benchmarks


def write_final_reports(repo: str | Path = ".", *, test_count: int) -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu4b"
    baseline = read_json(output / "baseline_manifest.json")
    environment = read_json(output / "environment_audit.json")
    attention = read_json(output / "attention_correctness.json")
    compile_result = read_json(output / "compile_qualification.json")
    benchmarks = write_gate_blocked_benchmarks(repo)
    historical_unchanged = all(
        file_sha256(repo / relative) == expected
        for relative, expected in read_json(output / "protocol.json")[
            "historical_frozen_inputs"
        ].items()
    )
    result = {
        "stage": "GPU-4B",
        "classification": "GPU4B_PARTIAL_VALIDATION",
        "environment": environment,
        "baseline": baseline,
        "attention_backends": {
            "manual": {"qualification": "PASS", "recommended": True},
            "sdpa_math": {
                "qualification": "CORRECTNESS_GATE_FAILED",
                "performance_eligible": False,
            },
            "sdpa_auto": {
                "qualification": "CORRECTNESS_GATE_FAILED",
                "actual_dispatch": environment["backend_qualification"][
                    "sdpa_auto_production_path"
                ]["selected"],
                "performance_eligible": False,
            },
            "sdpa_flash": {
                "qualification": "FLASH_BACKEND_NOT_AVAILABLE",
                "performance_eligible": False,
            },
        },
        "attention_correctness": attention,
        "attention_benchmarks": benchmarks,
        "compile_modes": compile_result["modes"],
        "compile_correctness": {
            "status": compile_result["correctness_status"],
            "checkpoint": compile_result["checkpoint_status"],
            "generation": compile_result["generation_status"],
            "graph_break_count": compile_result["graph_break_count"],
            "recompile_count": compile_result["recompile_count"],
        },
        "compile_benchmarks": {
            "status": compile_result["performance_status"],
            "uncached_prequalification_probe": compile_result["uncached_prequalification_probe"],
            "cold_qualification": compile_result["modes"],
            "steady_state": None,
            "end_to_end": None,
        },
        "final_candidate": {
            "attention_backend": "manual",
            "execution": "eager",
            "reason": "only configuration retaining the complete frozen correctness baseline",
        },
        "correctness_regressions": [
            "SDPA Math and Auto exceeded frozen BF16 allclose tolerance for full logits.",
            "Selected tied-embedding gradients exceeded the frozen max-absolute threshold.",
            "Single AdamW updates crossed the frozen per-element delta threshold on some parameters.",
        ],
        "performance_findings": [
            "No SDPA performance result was accepted because no optimized Attention backend passed B2.",
            "No torch.compile performance result was accepted because default and reduce-overhead failed with TritonMissing.",
            "GPU-4A Manual Eager remains the formal reference: 48.9 seconds, 12,188 input tokens/s, 1003.6/1318.0 MiB allocated/reserved peak.",
        ],
        "limitations": [
            "Current Windows PyTorch build has no built-in Flash Attention kernel.",
            "Current environment has no working Triton installation for Inductor CUDA compilation.",
            "Correctness-first gating intentionally prevents sequence-scaling and real-SFT speed claims.",
        ],
        "next_stage_recommendations": [
            "GPU-4C should investigate reproducible Windows kernel/compiler support as an isolated environment study, without changing this baseline.",
            "If a qualified software stack becomes available, rerun the frozen B2 tolerances before any performance benchmark.",
            "Keep Manual Eager as the production default in the current environment.",
        ],
        "tests": {"status": "PASS", "count": test_count},
        "historical_artifacts_preserved": historical_unchanged,
        "artifacts": {
            "environment": "artifacts/gpu4b/ENVIRONMENT_AND_BACKEND_AUDIT.md",
            "implementation": "artifacts/gpu4b/ATTENTION_IMPLEMENTATION_AUDIT.md",
            "attention_correctness": "artifacts/gpu4b/ATTENTION_CORRECTNESS.md",
            "attention_benchmark": "artifacts/gpu4b/ATTENTION_BENCHMARK.md",
            "compile_correctness": "artifacts/gpu4b/COMPILE_CORRECTNESS.md",
            "compile_benchmark": "artifacts/gpu4b/COMPILE_BENCHMARK.md",
            "comparison": "artifacts/gpu4b/FINAL_OPTIMIZATION_COMPARISON.md",
            "final": "artifacts/gpu4b/GPU4B_FINAL_REPORT.md",
        },
    }
    write_json(output / "gpu4b_result.json", result)
    (output / "FINAL_OPTIMIZATION_COMPARISON.md").write_text(
        """# Final optimization comparison

| Configuration | Correctness | Performance evidence | Recommendation |
|---|---|---|---|
| Manual Eager | GPU-4A full baseline retained | 48.9s formal SFT; 12,188 input tokens/s | Default |
| SDPA Math Eager | FP32 passed; frozen BF16/gradient/update gate failed | Not benchmarked after gate failure | Do not enable |
| SDPA Auto Eager | Efficient Attention dispatched; frozen BF16/gradient/update gate failed | Not benchmarked after gate failure | Do not enable |
| SDPA Flash Eager | Kernel unavailable | None | Not available |
| Manual + compile | Inductor failed with TritonMissing | No compiled steady state | Not available |
| SDPA + compile | Preconditions not met | None | Not run |

Speedups are intentionally not reported: adding percentages from missing or correctness-ineligible runs would be invalid. The current recommended configuration remains Manual Eager.
""",
        encoding="utf-8",
    )
    (output / "GPU4B_FINAL_REPORT.md").write_text(
        f"""# GPU-4B final report

Classification: `GPU4B_PARTIAL_VALIDATION`

The environment is Windows with Python {environment["python"]}, PyTorch {environment["pytorch"]}, CUDA runtime {environment["cuda_runtime"]}, driver {environment["cuda_driver"]}, and an RTX 5060 Laptop GPU (compute capability 12.0, BF16 supported). No software component was installed or upgraded.

The production Attention audit confirmed Manual GQA 8/4 with RoPE-before-attention, `1/sqrt(64)` scaling, causal plus key-padding masking, FP32 softmax, dropout semantics, and unchanged Q/K/V/O parameter shapes. GPU-4B preserved Manual as the historical default and added explicit experimental backends.

SDPA Math passed the strict FP32 reference: maximum logits error {attention["fp32"]["sdpa_math"]["logits"]["max_abs_error"]:.3g}, maximum first-layer Attention-output error {attention["fp32"]["sdpa_math"]["attention_output"]["max_abs_error"]:.3g}, and identical loss. In BF16, Math and Auto retained small mean logits errors and identical greedy answers/EOS behavior, compatible checkpoints, finite gradients, and matching 10-step improvement direction. They nevertheless exceeded the pre-frozen full-logit allclose, tied-embedding gradient, and one-step update thresholds, so both failed B2 and were excluded from performance testing.

Forced built-in Flash returned `FLASH_BACKEND_NOT_AVAILABLE`; the PyTorch build reports no available Flash kernel. Auto with explicit equivalent K/V expansion actually dispatched Efficient Attention, verified from operator traces, but that does not override its correctness failure.

Accordingly, 256/512/768/1024 attention-only and full-model benchmarks, as well as the 300-step real-SFT replay, are explicitly `NOT_RUN_CORRECTNESS_GATE_FAILED`. GPU-4A Manual Eager remains the only accepted real workload result: about 48.9 seconds, 12,188 input tokens/s, and 1003.6/1318.0 MiB peak allocated/reserved memory. Generation/T1/T2 regression testing for optimized candidates was not expanded to the full 384 records because the earlier numerical gate already failed; the fixed probes retained identical token IDs, answers, format, and EOS.

`torch.compile` exists at the API level, as do Dynamo and Inductor, but both `default` and `reduce-overhead` failed actual CUDA compilation with `TritonMissing`. The first uncached default attempt cost {compile_result["uncached_prequalification_probe"]["first_call_wall_seconds"]:.3f}s; the formal repeat then failed after {compile_result["modes"]["default"]["first_call_wall_seconds"]:.3f}s with caches already populated, while `reduce-overhead` failed after {compile_result["modes"]["reduce-overhead"]["first_call_wall_seconds"]:.3f}s. `max-autotune` was skipped because neither prerequisite mode worked. No eager fallback was mislabeled, so there is no steady-state or end-to-end compile speedup and no compiled checkpoint claim.

The recommended current configuration is `manual + eager`. SDPA Math/Auto should not be enabled under the frozen acceptance contract, Flash is unavailable, and compile is unsupported by the unchanged environment. Historical GPU-4A artifacts and checkpoints remain unchanged: `{historical_unchanged}`.

GPU-4C should isolate reproducible Windows fused-kernel and compiler-stack qualification. If a supported build is introduced in a separately controlled environment, rerun the frozen correctness gates before measuring performance; do not mix an environment upgrade with the current GPU-4B baseline.
""",
        encoding="utf-8",
    )
    return result
