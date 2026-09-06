from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _command(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, capture_output=True, text=True, check=False)
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 else f"ERROR({completed.returncode}): {output}"


def _memory(device: torch.device) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "free_mib": free_bytes / 1024**2,
        "total_mib": total_bytes / 1024**2,
    }


def _run_kernel_tests(device: torch.device, bf16_supported: bool) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    results: dict[str, Any] = {}

    allocation = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize(device)
    results["allocation"] = {
        "status": "PASS",
        "shape": list(allocation.shape),
        "finite": bool(torch.isfinite(allocation).all()),
    }

    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    product = left @ right
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(product).all()):
        raise FloatingPointError("CUDA matmul returned non-finite values")
    results["matmul"] = {"status": "PASS", "checksum": float(product.float().mean())}

    network = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 16),
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    inputs = torch.randn(64, 256, device=device)
    targets = torch.randn(64, 16, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(network(inputs), targets)
    loss.backward()
    gradients_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in network.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    torch.cuda.synchronize(device)
    if not gradients_finite:
        raise FloatingPointError("CUDA backward returned non-finite gradients")
    results["backward"] = {
        "status": "PASS",
        "loss": float(loss.detach()),
        "gradients_finite": gradients_finite,
    }

    if bf16_supported:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bf16_loss = torch.nn.functional.mse_loss(network(inputs), targets)
        bf16_loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if not bool(torch.isfinite(bf16_loss)):
            raise FloatingPointError("BF16 autocast returned a non-finite loss")
        results["bf16"] = {"status": "PASS", "loss": float(bf16_loss.detach())}
    else:
        results["bf16"] = {"status": "UNSUPPORTED"}

    results["memory"] = _memory(device)
    if results["memory"]["peak_allocated_mib"] <= 0:
        raise RuntimeError("CUDA memory accounting returned a zero peak")
    return results


def collect_torch_environment(path: str | Path) -> dict[str, Any]:
    from torch.utils import collect_env

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "torch.utils.collect_env"],
        capture_output=True,
        text=True,
        check=False,
    )
    content = completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")
    status: dict[str, Any] = {
        "direct_command_exit_code": completed.returncode,
        "fallback_used": False,
    }
    if completed.returncode != 0:
        status["fallback_used"] = True

        def tolerant_run(command: str | list[str]):
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=isinstance(command, str),
            )
            stdout, stderr = process.communicate()

            def decode(value: bytes) -> str:
                try:
                    return value.decode("utf-8").strip()
                except UnicodeDecodeError:
                    return value.decode("oem", errors="replace").strip()

            return (
                process.returncode,
                decode(stdout),
                decode(stderr),
            )

        original_run = collect_env.run
        collect_env.run = tolerant_run
        try:
            fallback = collect_env.pretty_str(collect_env.get_env_info())
        finally:
            collect_env.run = original_run
        content += "\n\nFALLBACK (OEM decode errors replaced):\n" + fallback
    destination.write_text(content, encoding="utf-8")
    status["path"] = str(destination)
    return status


def qualify_cuda_environment(
    output_path: str | Path,
    collect_env_path: str | Path,
    *,
    expected_capability: tuple[int, int] = (12, 0),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "uv_version": _command(["uv", "--version"]),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "cuda_toolkit": _command(["nvcc", "--version"]),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "expected_compute_capability": list(expected_capability),
        "qualification": "FAIL",
    }
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA PyTorch device is unavailable")
        if torch.version.cuda is None:
            raise RuntimeError("PyTorch is a CPU-only build")
        runtime = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
        if runtime < (12, 8):
            raise RuntimeError(f"CUDA runtime {torch.version.cuda} is below Blackwell minimum 12.8")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "compute_capability": list(capability),
                "total_vram_mib": properties.total_memory / 1024**2,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "torch_cuda_arch_list": torch.cuda.get_arch_list(),
            }
        )
        if capability != expected_capability:
            raise RuntimeError(
                f"expected compute capability {expected_capability}, detected {capability}"
            )
        result["kernel_tests"] = _run_kernel_tests(device, result["bf16_supported"])
        result["qualification"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["collect_env"] = collect_torch_environment(collect_env_path)
    except Exception as exc:
        result["collect_env"] = {"error": f"{type(exc).__name__}: {exc}"}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
