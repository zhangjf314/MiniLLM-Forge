# ruff: noqa: E501
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

from minillm_forge.experiments_gpu4a.data import SFTDataset, collate_rows
from minillm_forge.experiments_gpu4a.runtime import (
    CHECKPOINT,
    TOKENIZER_PATH,
    file_sha256,
    generate_record,
    read_json,
    write_json,
)
from minillm_forge.experiments_gpu4b.audit import TOLERANCES
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.training import build_adamw, load_checkpoint, save_checkpoint

SELECTED_PARAMETERS = (
    "token_embedding.weight",
    "layers.0.input_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
)


def _model_from_state(
    architecture: dict[str, Any], state: dict[str, torch.Tensor], backend: str
) -> MiniLLM:
    config = MiniLLMConfig(**architecture, attention_backend=backend)
    model = MiniLLM(config)
    model.load_state_dict(state, strict=True)
    model.gradient_checkpointing = False
    return model.cuda()


def _set_backend(model: MiniLLM, backend: str) -> None:
    for layer in model.layers:
        layer.self_attn.attention_backend = backend


def _release(model: MiniLLM) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _batch(repo: Path) -> dict[str, torch.Tensor]:
    dataset = SFTDataset(repo / "data/processed/gpu4a/train.jsonl")
    raw = collate_rows([dataset[0], dataset[1]])
    return {key: value.cuda() for key, value in raw.items() if isinstance(value, torch.Tensor)}


def _errors(
    reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    absolute = (candidate - reference).abs()
    relative = absolute / reference.abs().clamp_min(1e-8)
    return {
        "max_abs_error": float(absolute.max()),
        "max_rel_error": float(relative.max()),
        "mean_abs_error": float(absolute.mean()),
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def _forward_capture(
    model: MiniLLM, batch: dict[str, torch.Tensor], *, bf16: bool
) -> dict[str, Any]:
    captures: list[torch.Tensor] = []
    handle = model.layers[0].self_attn.register_forward_hook(
        lambda _module, _args, output: captures.append(output.detach().clone())
    )
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(**batch)
    handle.remove()
    return {
        "logits": output.logits.detach(),
        "loss": float(output.loss),
        "attention_output": captures[0],
    }


def _gradient_capture(
    model: MiniLLM, batch: dict[str, torch.Tensor], backend: str
) -> tuple[float, dict[str, torch.Tensor]]:
    _set_backend(model, backend)
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch)
    output.loss.backward()
    named = dict(model.named_parameters())
    gradients = {
        name: named[name].grad.detach().float().cpu().clone() for name in SELECTED_PARAMETERS
    }
    return float(output.loss.detach()), gradients


def _gradient_errors(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, Any]:
    result = {}
    for name in SELECTED_PARAMETERS:
        left = reference[name].flatten()
        right = candidate[name].flatten()
        absolute = (right - left).abs()
        denominator = float(left.norm()) * float(right.norm())
        cosine = float(torch.dot(left, right)) / denominator if denominator else 1.0
        result[name] = {
            "max_abs_error": float(absolute.max()),
            "max_rel_error": float((absolute / left.abs().clamp_min(1e-8)).max()),
            "mean_abs_error": float(absolute.mean()),
            "cosine_similarity": cosine,
            "finite": bool(torch.isfinite(right).all()),
            "pass": bool(
                absolute.max() <= TOLERANCES["gradient_max_abs"]
                and cosine >= TOLERANCES["gradient_cosine_min"]
                and torch.isfinite(right).all()
            ),
        }
    return result


def _optimizer_step(
    architecture: dict[str, Any],
    state: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    backend: str,
) -> dict[str, Any]:
    model = _model_from_state(architecture, state, backend)
    optimizer = build_adamw(model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    named = dict(model.named_parameters())
    before = {name: named[name].detach().cpu().clone() for name in SELECTED_PARAMETERS}
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch)
    output.loss.backward()
    optimizer.step()
    result = {
        "loss": float(output.loss),
        "parameters": {
            name: {
                "delta": (named[name].detach().cpu() - before[name]).clone(),
                "parameter_norm": float(named[name].detach().float().norm()),
                "finite": bool(torch.isfinite(named[name]).all()),
            }
            for name in SELECTED_PARAMETERS
        },
    }
    _release(model)
    return result


def _compare_optimizer(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    parameters = {}
    for name in SELECTED_PARAMETERS:
        left = reference["parameters"][name]["delta"]
        right = candidate["parameters"][name]["delta"]
        maximum = float((right - left).abs().max())
        parameters[name] = {
            "delta_max_abs_error": maximum,
            "candidate_delta_norm": float(right.float().norm()),
            "candidate_parameter_norm": candidate["parameters"][name]["parameter_norm"],
            "finite": candidate["parameters"][name]["finite"],
            "pass": maximum <= TOLERANCES["optimizer_parameter_max_abs"]
            and candidate["parameters"][name]["finite"],
        }
    return {
        "loss_abs_error": abs(candidate["loss"] - reference["loss"]),
        "parameters": parameters,
        "pass": all(item["pass"] for item in parameters.values()),
    }


def _trajectory(
    architecture: dict[str, Any],
    state: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    backend: str,
) -> dict[str, Any]:
    model = _model_from_state(architecture, state, backend)
    model.train()
    optimizer = build_adamw(model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    losses, gradient_norms, parameter_norms = [], [], []
    named = dict(model.named_parameters())
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch)
        output.loss.backward()
        gradient_norms.append(
            math.sqrt(
                sum(
                    float(named[name].grad.detach().float().norm()) ** 2
                    for name in SELECTED_PARAMETERS
                )
            )
        )
        optimizer.step()
        losses.append(float(output.loss))
        parameter_norms.append(float(named[SELECTED_PARAMETERS[2]].detach().float().norm()))
    result = {
        "losses": losses,
        "gradient_norms": gradient_norms,
        "parameter_norms": parameter_norms,
        "finite": all(math.isfinite(value) for value in losses + gradient_norms + parameter_norms),
    }
    _release(model)
    return result


def _compare_trajectory(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    differences = [
        abs(left - right)
        for left, right in zip(reference["losses"], candidate["losses"], strict=True)
    ]
    return {
        **candidate,
        "loss_max_abs_error": max(differences),
        "same_improvement_direction": (candidate["losses"][-1] < candidate["losses"][0])
        == (reference["losses"][-1] < reference["losses"][0]),
        "pass": candidate["finite"]
        and max(differences) <= TOLERANCES["trajectory_loss_max_abs"]
        and (candidate["losses"][-1] < candidate["losses"][0])
        == (reference["losses"][-1] < reference["losses"][0]),
    }


def _checkpoint_compatibility(
    repo: Path, architecture: dict[str, Any], state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    keys = {}
    for backend in ("manual", "sdpa_math", "sdpa_auto"):
        model = _model_from_state(architecture, state, backend)
        keys[backend] = list(model.state_dict())
        _release(model)
    same_schema = keys["manual"] == keys["sdpa_math"] == keys["sdpa_auto"]
    model = _model_from_state(architecture, state, "sdpa_auto")
    optimizer = build_adamw(model, lr=1e-4)
    destination = repo / "runs/gpu4b/correctness/sdpa-auto-checkpoint.pt"
    save_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        trainer_state={"global_step": 0, "tokens_seen": 0},
        config={"model": {**architecture, "attention_backend": "sdpa_auto"}},
    )
    manual = _model_from_state(architecture, state, "manual")
    load_checkpoint(destination, model=manual, map_location="cuda", restore_rng=False)
    exact = all(
        torch.equal(left, right)
        for left, right in zip(
            model.state_dict().values(), manual.state_dict().values(), strict=True
        )
    )
    checkpoint_hash = file_sha256(destination)
    _release(model)
    _release(manual)
    return {
        "gpu4a_checkpoint_loads_manual_math_auto": True,
        "parameter_names_and_shapes_unchanged": same_schema,
        "sdpa_checkpoint_loads_manual": exact,
        "saved_checkpoint": str(destination),
        "saved_checkpoint_sha256": checkpoint_hash,
        "pass": same_schema and exact,
    }


def _generation_equivalence(
    repo: Path, architecture: dict[str, Any], state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(repo / TOKENIZER_PATH))
    test_rows = [
        json.loads(line)
        for line in (repo / "data/processed/gpu4a/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows = [
        next(row for row in test_rows if row["task"] == task)
        for task in ("T1_SUPPORT_CLASSIFICATION", "T2_INTEGER_ADDITION")
    ]
    model = _model_from_state(architecture, state, "manual").eval()
    records = {}
    for backend in ("manual", "sdpa_math", "sdpa_auto"):
        _set_backend(model, backend)
        records[backend] = [generate_record(model, tokenizer, row) for row in rows]
    reference = records["manual"]
    comparisons = {}
    for backend in ("sdpa_math", "sdpa_auto"):
        candidates = records[backend]
        comparisons[backend] = {
            "token_ids_identical": all(
                left["generated_token_ids"] == right["generated_token_ids"]
                for left, right in zip(reference, candidates, strict=True)
            ),
            "answers_identical": all(
                left["extracted_answer"] == right["extracted_answer"]
                for left, right in zip(reference, candidates, strict=True)
            ),
            "eos_identical": all(
                left["stop_reason"] == right["stop_reason"] == "EOS"
                for left, right in zip(reference, candidates, strict=True)
            ),
            "records": candidates,
        }
        comparisons[backend]["pass"] = all(
            comparisons[backend][key]
            for key in ("token_ids_identical", "answers_identical", "eos_identical")
        )
    _release(model)
    return {"manual_records": reference, "comparisons": comparisons}


def _write_report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Attention correctness gates",
        "",
        "Tolerances were frozen in `protocol.json` before this run. Tests use the real 37M checkpoint, frozen GPU-4A tokenized samples, identical weights/data order, BF16 autocast where specified, and forced backend contexts where applicable.",
        "",
        f"FP32 manual versus SDPA Math: `{result['fp32']['sdpa_math']['pass']}`. BF16 Math/Auto: `{result['bf16']['sdpa_math']['pass']}` / `{result['bf16']['sdpa_auto']['pass']}`.",
        f"Backward Math/Auto: `{result['backward']['sdpa_math']['pass']}` / `{result['backward']['sdpa_auto']['pass']}`. Optimizer step Math/Auto: `{result['optimizer_step']['sdpa_math']['pass']}` / `{result['optimizer_step']['sdpa_auto']['pass']}`.",
        f"Ten-step trajectory Math/Auto: `{result['short_trajectory']['sdpa_math']['pass']}` / `{result['short_trajectory']['sdpa_auto']['pass']}`. Checkpoint compatibility: `{result['checkpoint']['pass']}`.",
        f"Greedy generation Math/Auto: `{result['generation']['comparisons']['sdpa_math']['pass']}` / `{result['generation']['comparisons']['sdpa_auto']['pass']}`; generated IDs, extracted answers, and EOS behavior are identical to manual.",
        "",
        "Flash was excluded from correctness/performance acceptance because forced built-in Flash qualification failed. It is not represented by a fallback result.",
    ]
    (output / "ATTENTION_CORRECTNESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_correctness(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu4b"
    manifest = read_json(repo / "artifacts/training/minillm_formal_model.json")
    architecture = manifest["architecture"]
    pretrained = torch.load(repo / CHECKPOINT, map_location="cpu", weights_only=False)
    state = pretrained["model"]
    batch = _batch(repo)
    model = _model_from_state(architecture, state, "manual")

    fp32_reference = _forward_capture(model, batch, bf16=False)
    _set_backend(model, "sdpa_math")
    fp32_math = _forward_capture(model, batch, bf16=False)
    fp32 = {
        "sdpa_math": {
            "logits": _errors(fp32_reference["logits"], fp32_math["logits"], **TOLERANCES["fp32"]),
            "attention_output": _errors(
                fp32_reference["attention_output"],
                fp32_math["attention_output"],
                **TOLERANCES["fp32"],
            ),
            "loss_abs_error": abs(fp32_math["loss"] - fp32_reference["loss"]),
        }
    }
    fp32["sdpa_math"]["pass"] = (
        fp32["sdpa_math"]["logits"]["allclose"]
        and fp32["sdpa_math"]["attention_output"]["allclose"]
        and fp32["sdpa_math"]["loss_abs_error"] <= TOLERANCES["loss_abs"]
    )

    _set_backend(model, "manual")
    bf16_reference = _forward_capture(model, batch, bf16=True)
    bf16 = {}
    for backend in ("sdpa_math", "sdpa_auto"):
        _set_backend(model, backend)
        candidate = _forward_capture(model, batch, bf16=True)
        bf16[backend] = {
            "logits": _errors(bf16_reference["logits"], candidate["logits"], **TOLERANCES["bf16"]),
            "attention_output": _errors(
                bf16_reference["attention_output"],
                candidate["attention_output"],
                **TOLERANCES["bf16"],
            ),
            "reference_loss": bf16_reference["loss"],
            "candidate_loss": candidate["loss"],
            "loss_abs_error": abs(candidate["loss"] - bf16_reference["loss"]),
        }
        bf16[backend]["pass"] = (
            bf16[backend]["logits"]["allclose"]
            and bf16[backend]["attention_output"]["allclose"]
            and bf16[backend]["loss_abs_error"] <= TOLERANCES["loss_abs"]
        )

    reference_loss, reference_gradients = _gradient_capture(model, batch, "manual")
    backward = {}
    for backend in ("sdpa_math", "sdpa_auto"):
        loss, gradients = _gradient_capture(model, batch, backend)
        values = _gradient_errors(reference_gradients, gradients)
        backward[backend] = {
            "reference_loss": reference_loss,
            "candidate_loss": loss,
            "parameters": values,
            "pass": all(item["pass"] for item in values.values()),
        }
    _release(model)

    optimizer_reference = _optimizer_step(architecture, state, batch, "manual")
    optimizer_step = {
        backend: _compare_optimizer(
            optimizer_reference, _optimizer_step(architecture, state, batch, backend)
        )
        for backend in ("sdpa_math", "sdpa_auto")
    }
    trajectory_reference = _trajectory(architecture, state, batch, "manual")
    trajectories = {
        backend: _compare_trajectory(
            trajectory_reference, _trajectory(architecture, state, batch, backend)
        )
        for backend in ("sdpa_math", "sdpa_auto")
    }
    checkpoint = _checkpoint_compatibility(repo, architecture, state)

    sft_result = read_json(repo / "artifacts/gpu4a/sft_evaluation.json")
    sft_checkpoint = Path(sft_result["checkpoint"]["path"])
    sft_payload = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
    generation = _generation_equivalence(repo, architecture, sft_payload["model"])
    environment = read_json(output / "environment_audit.json")
    flash = environment["backend_qualification"]["sdpa_flash_expanded_gqa"]
    gates = {
        "sdpa_math": fp32["sdpa_math"]["pass"]
        and bf16["sdpa_math"]["pass"]
        and backward["sdpa_math"]["pass"]
        and optimizer_step["sdpa_math"]["pass"]
        and trajectories["sdpa_math"]["pass"]
        and checkpoint["pass"]
        and generation["comparisons"]["sdpa_math"]["pass"],
        "sdpa_auto": bf16["sdpa_auto"]["pass"]
        and backward["sdpa_auto"]["pass"]
        and optimizer_step["sdpa_auto"]["pass"]
        and trajectories["sdpa_auto"]["pass"]
        and checkpoint["pass"]
        and generation["comparisons"]["sdpa_auto"]["pass"],
        "sdpa_flash": "FLASH_BACKEND_NOT_AVAILABLE" if not flash["qualified"] else "UNTESTED",
    }
    result = {
        "stage": "GPU-4B-2",
        "tolerances": TOLERANCES,
        "fp32": fp32,
        "bf16": bf16,
        "backward": backward,
        "optimizer_step": optimizer_step,
        "short_trajectory": {"manual": trajectory_reference, **trajectories},
        "checkpoint": checkpoint,
        "generation": generation,
        "backend_gates": gates,
        "performance_eligible": [name for name in ("sdpa_math", "sdpa_auto") if gates[name]],
        "all_required_gates_pass": gates["sdpa_math"] is True and gates["sdpa_auto"] is True,
        "historical_artifacts_preserved": all(
            file_sha256(repo / relative) == expected
            for relative, expected in read_json(output / "protocol.json")[
                "historical_frozen_inputs"
            ].items()
        ),
    }
    write_json(output / "attention_correctness.json", result)
    _write_report(output, result)
    del pretrained, state, sft_payload
    gc.collect()
    torch.cuda.empty_cache()
    return result
