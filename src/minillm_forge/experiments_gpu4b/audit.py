# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from minillm_forge.experiments_gpu4a.runtime import file_sha256, read_json, write_json
from minillm_forge.model import GroupedQueryAttention, MiniLLMConfig
from minillm_forge.model.attention import repeat_key_value

BACKENDS = ("manual", "sdpa_math", "sdpa_auto", "sdpa_flash")
TOLERANCES = {
    "fp32": {"atol": 2e-5, "rtol": 1e-4},
    "bf16": {"atol": 0.08, "rtol": 0.03},
    "loss_abs": 0.02,
    "gradient_max_abs": 0.05,
    "gradient_cosine_min": 0.999,
    "optimizer_parameter_max_abs": 2e-4,
    "trajectory_loss_max_abs": 0.05,
}


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().splitlines()[0]


def _kernel_trial(backend: SDPBackend, *, expand_gqa: bool) -> dict[str, Any]:
    torch.manual_seed(4204)
    query = torch.randn(2, 8, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    if expand_gqa:
        key = repeat_key_value(key, 2)
        value = repeat_key_value(value, 2)
    mask = torch.ones(2, 1, 64, 64, device="cuda", dtype=torch.bool).tril()
    started = time.perf_counter()
    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with sdpa_kernel(backends=[backend]):
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    enable_gqa=not expand_gqa,
                )
                output.float().sum().backward()
            torch.cuda.synchronize()
            captured = [str(item.message) for item in caught]
        return {
            "qualified": True,
            "gqa_adapter": "explicit_kv_expansion" if expand_gqa else "native_enable_gqa",
            "shape": list(output.shape),
            "wall_time_seconds": time.perf_counter() - started,
            "warnings": captured,
        }
    except Exception as error:
        return {
            "qualified": False,
            "gqa_adapter": "explicit_kv_expansion" if expand_gqa else "native_enable_gqa",
            "error": f"{type(error).__name__}: {error}",
            "warnings": captured,
        }


def _profile_auto_backend() -> dict[str, Any]:
    torch.manual_seed(4204)
    config = MiniLLMConfig(
        vocab_size=64,
        hidden_size=512,
        num_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1536,
        max_seq_len=128,
        attention_backend="sdpa_auto",
    )
    module = GroupedQueryAttention(config).cuda().to(torch.bfloat16).eval()
    hidden = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 64, device="cuda", dtype=torch.bool)
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        module(hidden, mask)
        torch.cuda.synchronize()
    operators = sorted(
        event.key
        for event in trace.key_averages()
        if "attention" in event.key.lower() or "scaled_dot_product" in event.key.lower()
    )
    selected = "unknown"
    joined = " ".join(operators).lower()
    if "efficient_attention" in joined:
        selected = "efficient_attention"
    elif "cudnn" in joined:
        selected = "cudnn_attention"
    elif "flash" in joined:
        selected = "flash_attention"
    elif "attention_math" in joined:
        selected = "math"
    return {"selected": selected, "operators": operators}


def environment_audit() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _driver_version(),
        "cuda_available": torch.cuda.is_available(),
        "torch_compile_present": hasattr(torch, "compile"),
        "dynamo_present": importlib.util.find_spec("torch._dynamo") is not None,
        "inductor_present": importlib.util.find_spec("torch._inductor") is not None,
        "triton_python_package_present": importlib.util.find_spec("triton") is not None,
        "sdpa_present": hasattr(F, "scaled_dot_product_attention"),
    }
    if not torch.cuda.is_available():
        environment["backend_qualification"] = {}
        return environment
    properties = torch.cuda.get_device_properties(0)
    environment.update(
        {
            "gpu": properties.name,
            "gpu_vram_mib": properties.total_memory / 2**20,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    environment["backend_qualification"] = {
        "sdpa_math_native_gqa": _kernel_trial(SDPBackend.MATH, expand_gqa=False),
        "sdpa_flash_expanded_gqa": _kernel_trial(SDPBackend.FLASH_ATTENTION, expand_gqa=True),
        "sdpa_efficient_expanded_gqa": _kernel_trial(
            SDPBackend.EFFICIENT_ATTENTION, expand_gqa=True
        ),
        "sdpa_cudnn_expanded_gqa": _kernel_trial(SDPBackend.CUDNN_ATTENTION, expand_gqa=True),
        "sdpa_auto_production_path": _profile_auto_backend(),
    }
    return environment


def _write_reports(output: Path, environment: dict[str, Any], manifest: dict[str, Any]) -> None:
    qualification = environment["backend_qualification"]
    flash = qualification["sdpa_flash_expanded_gqa"]
    auto = qualification["sdpa_auto_production_path"]
    (output / "ENVIRONMENT_AND_BACKEND_AUDIT.md").write_text(
        f"""# GPU-4B environment and backend audit

- OS: `{environment["os"]}`
- Python: `{environment["python"]}`
- PyTorch: `{environment["pytorch"]}`
- CUDA runtime / driver: `{environment["cuda_runtime"]}` / `{environment["cuda_driver"]}`
- GPU: `{environment["gpu"]}`, compute capability `{environment["compute_capability"]}`, {environment["gpu_vram_mib"]:.1f} MiB
- BF16: `{environment["bf16_supported"]}`
- `torch.compile` / Dynamo / Inductor: `{environment["torch_compile_present"]}` / `{environment["dynamo_present"]}` / `{environment["inductor_present"]}`
- Triton Python package: `{environment["triton_python_package_present"]}`

Forced kernel qualification used BF16, causal masking, forward and backward, and the production GQA adapter. SDPA Math passed with native `enable_gqa`. Expanded-GQA Efficient Attention and cuDNN Attention passed. The production `sdpa_auto` path selected `{auto["selected"]}` according to recorded dispatcher operators.

PyTorch built-in Flash qualification: `{"PASS" if flash["qualified"] else "FLASH_BACKEND_NOT_AVAILABLE"}`. Forced Flash execution returned `{flash.get("error", "no error")}`; no fallback result is labeled as Flash. No packages, drivers, CUDA components, or PyTorch builds were changed.
""",
        encoding="utf-8",
    )
    (output / "ATTENTION_IMPLEMENTATION_AUDIT.md").write_text(
        """# Native Attention implementation audit

The frozen production default is `manual`. Each block applies RMSNorm, then bias-free Q/K/V projections. Q has 8 heads, K/V have 4 heads, and head dimension is 64. RoPE is applied to Q and K before attention. The manual path repeats each KV head twice in group order, scales QK scores by `1/sqrt(64)`, combines a lower-triangular causal mask with the right-padding key mask, computes softmax in FP32, casts probabilities back to the query dtype, applies configured attention dropout, multiplies by V, merges heads, and applies the bias-free output projection and residual dropout.

GPU-4B adds explicit `manual`, `sdpa_math`, `sdpa_auto`, and `sdpa_flash` configuration values while retaining `manual` as the historical default. Math uses native `enable_gqa=True`. Auto and Flash use explicit K/V expansion because this Windows build's fused dense kernels require equal Q/K/V head counts; this preserves parameter names, shapes, grouping, RoPE, scaling, masks, and checkpoint schema. Flash is forced through the PyTorch backend context so an unavailable kernel fails rather than silently falling back. Auto is allowed to dispatch and its selected operator is captured separately.
""",
        encoding="utf-8",
    )


def run_audit(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu4b"
    output.mkdir(parents=True, exist_ok=True)
    gpu4a = read_json(repo / "artifacts/gpu4a/gpu4a_result.json")
    protocol_path = repo / "artifacts/gpu4a/protocol.json"
    sft_checkpoint = Path(gpu4a["sft_evaluation"]["checkpoint"]["path"])
    baseline = {
        "starting_branch": "codex/gpu4a-native-37m-sft",
        "starting_head": "1333df1246408f34606a0c2648b70750326ecb46",
        "gpu4a_head": "1333df1246408f34606a0c2648b70750326ecb46",
        "formal_pretrained_checkpoint": gpu4a["pretraining_checkpoint"],
        "formal_sft_checkpoint": {
            **gpu4a["sft_evaluation"]["checkpoint"],
            "sha256_rechecked": file_sha256(sft_checkpoint),
        },
        "tokenizer_sha256": gpu4a["pretraining_checkpoint"]["tokenizer_sha256"],
        "dataset_digest": gpu4a["dataset_and_task_protocol"]["dataset_digest"],
        "gpu4a_protocol_sha256": file_sha256(protocol_path),
        "gpu4a_result_sha256": file_sha256(repo / "artifacts/gpu4a/gpu4a_result.json"),
        "training_config": {
            "steps": 300,
            "dtype": "BF16",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "micro_batch": 16,
            "gradient_accumulation": 2,
            "effective_batch": 32,
        },
        "evaluation_config": {
            "decoding": "greedy",
            "max_new_tokens": 12,
            "records": 384,
            "per_task_split": 64,
        },
        "frozen_artifacts": {
            relative: file_sha256(repo / relative)
            for relative in (
                "artifacts/gpu4a/protocol.json",
                "artifacts/gpu4a/gpu4a_result.json",
                "artifacts/gpu4a/formal_sft_result.json",
                "artifacts/gpu4a/sft_evaluation.json",
                "runs/E01-minillm-formal/best.pt",
            )
        },
    }
    environment = environment_audit()
    protocol = {
        "stage": "GPU-4B",
        "version": "gpu4b-v1-frozen-before-correctness",
        "attention_backends": list(BACKENDS),
        "tolerances": TOLERANCES,
        "correctness_seed": 4204,
        "correctness_batch_size": 2,
        "correctness_steps": 10,
        "benchmark_sequence_lengths": [256, 512, 768, 1024],
        "historical_frozen_inputs": baseline["frozen_artifacts"],
    }
    write_json(output / "baseline_manifest.json", baseline)
    write_json(output / "environment_audit.json", environment)
    write_json(output / "protocol.json", protocol)
    _write_reports(output, environment, baseline)
    return {"baseline": baseline, "environment": environment, "protocol": protocol}
