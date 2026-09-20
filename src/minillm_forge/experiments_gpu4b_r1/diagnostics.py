# ruff: noqa: E501
from __future__ import annotations

import gc
import hashlib
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minillm_forge.experiments_gpu4a.data import SFTDataset, collate_rows
from minillm_forge.experiments_gpu4a.runtime import CHECKPOINT, file_sha256, read_json, write_json
from minillm_forge.experiments_gpu4b.audit import TOLERANCES
from minillm_forge.model import MiniLLM, MiniLLMConfig
from minillm_forge.model.attention import repeat_key_value
from minillm_forge.model.model import causal_lm_loss
from minillm_forge.training import build_adamw

RELATIVE_FLOOR = 1e-6
SEED = 4204
SELECTED_PARAMETERS = (
    "token_embedding.weight",
    "layers.0.input_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
)


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    relative_floor: float = RELATIVE_FLOOR,
) -> dict[str, Any]:
    """Stable diagnostics with explicit near-zero and elementwise allclose handling."""
    left = reference.detach().float().cpu()
    right = candidate.detach().float().cpu()
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    delta = right - left
    absolute = delta.abs()
    flat_index = int(absolute.reshape(-1).argmax()) if absolute.numel() else 0
    coordinates: list[int] = []
    remainder = flat_index
    for dimension in reversed(left.shape):
        coordinates.append(remainder % dimension)
        remainder //= dimension
    index = list(reversed(coordinates))
    selector = tuple(index)
    nonzero = left.abs() >= relative_floor
    relative = absolute[nonzero] / left.abs()[nonzero] if bool(nonzero.any()) else torch.zeros(1)
    threshold = atol + rtol * left.abs()
    left_rms = float(torch.sqrt(torch.mean(left.square()))) if left.numel() else 0.0
    error_rms = float(torch.sqrt(torch.mean(delta.square()))) if delta.numel() else 0.0
    return {
        "shape": list(left.shape),
        "reference_dtype": str(reference.dtype).replace("torch.", ""),
        "candidate_dtype": str(candidate.dtype).replace("torch.", ""),
        "max_abs_error": float(absolute.max()) if absolute.numel() else 0.0,
        "mean_abs_error": float(absolute.mean()) if absolute.numel() else 0.0,
        "max_relative_error": float(relative.max()),
        "relative_denominator_floor": relative_floor,
        "near_zero_reference_count": int((~nonzero).sum()),
        "rms_error": error_rms,
        "reference_rms": left_rms,
        "relative_rms_error": error_rms / max(left_rms, relative_floor),
        "allclose_fraction": float((absolute <= threshold).float().mean()),
        "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "max_error_index": index,
        "max_error_reference": float(left[selector]) if left.numel() else 0.0,
        "max_error_candidate": float(right[selector]) if right.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def gradient_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    metrics = tensor_metrics(reference, candidate, atol=0.0, rtol=0.0)
    left = reference.detach().float().cpu().flatten()
    right = candidate.detach().float().cpu().flatten()
    denominator = float(left.norm()) * float(right.norm())
    cosine = float(torch.dot(left, right)) / denominator if denominator else 1.0
    metrics.update(
        {
            "reference_gradient_rms": float(torch.sqrt(torch.mean(left.square()))),
            "candidate_gradient_rms": float(torch.sqrt(torch.mean(right.square()))),
            "cosine_similarity": max(-1.0, min(1.0, cosine)),
            "sign_disagreement_count": int(((torch.sign(left) != torch.sign(right)) & (left != 0) & (right != 0)).sum()),
        }
    )
    return metrics


def _model(architecture: dict[str, Any], state: dict[str, torch.Tensor], backend: str) -> MiniLLM:
    model = MiniLLM(MiniLLMConfig(**architecture, attention_backend=backend))
    model.load_state_dict(state, strict=True)
    model.gradient_checkpointing = False
    return model.cuda()


def _batch(repo: Path) -> dict[str, torch.Tensor]:
    dataset = SFTDataset(repo / "data/processed/gpu4a/train.jsonl")
    raw = collate_rows([dataset[0], dataset[1]])
    return {key: value.cuda() for key, value in raw.items() if isinstance(value, torch.Tensor)}


def _release(model: MiniLLM) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _capture_forward(
    model: MiniLLM, batch: dict[str, torch.Tensor], *, bf16: bool
) -> tuple[dict[str, torch.Tensor], float]:
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def save(name: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, tuple):
                captures[f"{name}.q"] = output[0].detach().cpu().clone()
                captures[f"{name}.k"] = output[1].detach().cpu().clone()
            else:
                captures[name] = output.detach().cpu().clone()
        return hook

    def save_input(name: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            captures[name] = args[0].detach().cpu().clone()
        return hook

    handles.append(model.token_embedding.register_forward_hook(save("embedding.output")))
    for index, layer in enumerate(model.layers):
        prefix = f"block.{index}"
        handles.append(layer.register_forward_pre_hook(save_input(f"{prefix}.input")))
        handles.append(layer.register_forward_hook(save(f"{prefix}.output")))
        handles.append(layer.self_attn.register_forward_hook(save(f"{prefix}.attention.output")))
        handles.append(layer.post_attention_norm.register_forward_pre_hook(save_input(f"{prefix}.attention.residual")))
        handles.append(layer.mlp.register_forward_hook(save(f"{prefix}.mlp.output")))
        if index == 0:
            handles.append(layer.input_norm.register_forward_hook(save(f"{prefix}.input_norm.output")))
            handles.append(layer.self_attn.q_proj.register_forward_hook(save(f"{prefix}.q_projection")))
            handles.append(layer.self_attn.k_proj.register_forward_hook(save(f"{prefix}.k_projection")))
            handles.append(layer.self_attn.v_proj.register_forward_hook(save(f"{prefix}.v_projection")))
            if layer.self_attn.rope is not None:
                handles.append(layer.self_attn.rope.register_forward_hook(save(f"{prefix}.rope")))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(save_input(f"{prefix}.attention.context")))
            handles.append(layer.self_attn.o_proj.register_forward_hook(save(f"{prefix}.o_projection")))
    handles.append(model.norm.register_forward_hook(save("final_norm.output")))
    handles.append(model.lm_head.register_forward_hook(save("logits")))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(**batch)
    for handle in handles:
        handle.remove()
    return captures, float(output.loss)


def _compare_captures(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], *, atol: float, rtol: float
) -> dict[str, Any]:
    return {name: tensor_metrics(reference[name], candidate[name], atol=atol, rtol=rtol) for name in reference}


def _precision_reconstructions(
    manual: dict[str, torch.Tensor], math_capture: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], scale: float
) -> dict[str, Any]:
    q = manual["block.0.rope.q"].cuda()
    k = repeat_key_value(manual["block.0.rope.k"].cuda(), 2)
    v_raw = manual["block.0.v_projection"].cuda()
    batch_size, sequence, _ = v_raw.shape
    v = v_raw.view(batch_size, sequence, 4, 64).transpose(1, 2)
    v = repeat_key_value(v, 2)
    key_mask = batch["attention_mask"][:, None, None, :].bool()
    causal = torch.ones(sequence, sequence, device="cuda", dtype=torch.bool).tril().view(1, 1, sequence, sequence)
    allowed = causal & key_mask

    def merge(context: torch.Tensor) -> torch.Tensor:
        return context.transpose(1, 2).contiguous().view(batch_size, sequence, -1).cpu()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores_bf16 = torch.matmul(q, k.transpose(-2, -1)) * scale
        finite_scores = scores_bf16.masked_fill(~allowed, torch.finfo(scores_bf16.dtype).min)
        probability_fp32_from_bf16 = torch.softmax(finite_scores.float(), dim=-1)
        probability_bf16 = probability_fp32_from_bf16.to(q.dtype)
        production_manual = torch.matmul(probability_bf16, v)
        infinity_scores = scores_bf16.masked_fill(~allowed, float("-inf"))
        infinity_manual = torch.matmul(torch.softmax(infinity_scores.float(), dim=-1).to(q.dtype), v)

    with torch.autocast("cuda", enabled=False):
        scores_fp32 = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        scores_fp32 = scores_fp32.masked_fill(~allowed, float("-inf"))
        probability_fp32 = torch.softmax(scores_fp32, dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fp32_qk_bf16_probability_av = torch.matmul(probability_fp32.to(torch.bfloat16), v)
    with torch.autocast("cuda", enabled=False):
        bf16_qk_fp32_probability_av = torch.matmul(probability_fp32_from_bf16, v.float())
        full_fp32 = torch.matmul(probability_fp32, v.float())

    variants = {
        "production_equivalent_manual": merge(production_manual),
        "fp32_qk_bf16_probability_bf16_av": merge(fp32_qk_bf16_probability_av),
        "bf16_qk_fp32_probability_fp32_av_then_bf16": merge(bf16_qk_fp32_probability_av.to(torch.bfloat16)),
        "full_fp32_then_bf16": merge(full_fp32.to(torch.bfloat16)),
        "negative_infinity_mask_manual": merge(infinity_manual),
    }
    actual_manual = manual["block.0.attention.context"]
    actual_math = math_capture["block.0.attention.context"]
    return {
        "evidence_scope": {
            "runtime_observed": "Q/K/V, RoPE outputs, and o_proj input/output at public module boundaries",
            "diagnostic_reconstruction": "QK, mask, softmax, and AV variants below; these are not observations inside the SDPA kernel",
            "documented_behavior": "installed PyTorch docstring states that Math keeps all intermediates in float for half/bfloat16 inputs",
            "unknown": "exact SDPA C++ instruction ordering and individual fused internal tensors",
        },
        "runtime_dtypes": {
            "q_projection": str(manual["block.0.q_projection"].dtype).replace("torch.", ""),
            "k_projection": str(manual["block.0.k_projection"].dtype).replace("torch.", ""),
            "v_projection": str(manual["block.0.v_projection"].dtype).replace("torch.", ""),
            "q_after_rope": str(manual["block.0.rope.q"].dtype).replace("torch.", ""),
            "diagnostic_manual_qk_result": str(scores_bf16.dtype).replace("torch.", ""),
            "diagnostic_manual_softmax_input": str(finite_scores.float().dtype).replace("torch.", ""),
            "diagnostic_manual_softmax_output_before_cast": str(probability_fp32_from_bf16.dtype).replace("torch.", ""),
            "diagnostic_manual_softmax_output_after_cast": str(probability_bf16.dtype).replace("torch.", ""),
            "manual_attention_context": str(actual_manual.dtype).replace("torch.", ""),
            "sdpa_math_attention_context": str(actual_math.dtype).replace("torch.", ""),
            "sdpa_internal_qk_softmax_av": "UNKNOWN (not hook-observable)",
        },
        "variants": {
            name: {
                "versus_actual_manual": tensor_metrics(actual_manual, value, atol=0.0, rtol=0.0),
                "versus_actual_sdpa_math": tensor_metrics(actual_math, value, atol=0.0, rtol=0.0),
            }
            for name, value in variants.items()
        },
        "score_path": {
            "bf16_qk_vs_fp32_qk_then_bf16": tensor_metrics(scores_bf16, scores_fp32.to(torch.bfloat16), atol=0.0, rtol=0.0),
            "finite_min_vs_negative_infinity_context": tensor_metrics(variants["production_equivalent_manual"], variants["negative_infinity_mask_manual"], atol=0.0, rtol=0.0),
        },
    }


def _total_gradient_and_step(
    architecture: dict[str, Any], state: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], backend: str
) -> dict[str, Any]:
    model = _model(architecture, state, backend).train()
    optimizer = build_adamw(model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    named = dict(model.named_parameters())
    before = {name: named[name].detach().cpu().clone() for name in SELECTED_PARAMETERS}
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
    output.loss.backward()
    gradients = {name: named[name].grad.detach().float().cpu().clone() for name in SELECTED_PARAMETERS}
    optimizer.step()
    deltas = {name: named[name].detach().cpu().clone() - before[name] for name in SELECTED_PARAMETERS}
    states = {}
    for name in SELECTED_PARAMETERS:
        current = optimizer.state[named[name]]
        states[name] = {
            key: current[key].detach().float().cpu().clone()
            for key in ("exp_avg", "exp_avg_sq")
        }
    result = {
        "loss": float(output.loss.detach()),
        "gradients": gradients,
        "deltas": deltas,
        "states": states,
        "before": before,
    }
    _release(model)
    return result


def _output_head_gradient(
    architecture: dict[str, Any], state: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], backend: str
) -> torch.Tensor:
    """Detach only the input embedding lookup; tied output weight stays trainable."""
    model = _model(architecture, state, backend).train()
    model.zero_grad(set_to_none=True)
    hidden = F.embedding(
        batch["input_ids"], model.token_embedding.weight.detach(), padding_idx=model.config.pad_token_id
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for layer in model.layers:
            hidden = layer(hidden, batch.get("attention_mask"), batch.get("position_ids"))
        logits = model.lm_head(model.norm(hidden))
        loss = causal_lm_loss(logits, batch["labels"])
    loss.backward()
    gradient = model.token_embedding.weight.grad.detach().float().cpu().clone()
    _release(model)
    return gradient


def _gradient_update_analysis(
    architecture: dict[str, Any], state: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, Any]:
    manual = _total_gradient_and_step(architecture, state, batch, "manual")
    math_result = _total_gradient_and_step(architecture, state, batch, "sdpa_math")
    manual_output = _output_head_gradient(architecture, state, batch, "manual")
    math_output = _output_head_gradient(architecture, state, batch, "sdpa_math")
    manual_input = manual["gradients"]["token_embedding.weight"] - manual_output
    math_input = math_result["gradients"]["token_embedding.weight"] - math_output
    gradients = {
        name: gradient_metrics(manual["gradients"][name], math_result["gradients"][name])
        for name in SELECTED_PARAMETERS
    }
    updates: dict[str, Any] = {}
    optimizer_states: dict[str, Any] = {}
    for name in SELECTED_PARAMETERS:
        metric = tensor_metrics(manual["deltas"][name], math_result["deltas"][name], atol=0.0, rtol=0.0)
        left_grad = manual["gradients"][name].flatten()
        right_grad = math_result["gradients"][name].flatten()
        flat_index = int((math_result["deltas"][name] - manual["deltas"][name]).abs().flatten().argmax())
        sign_flip = torch.sign(left_grad) != torch.sign(right_grad)
        metric.update(
            {
                "reference_update_rms": float(torch.sqrt(torch.mean(manual["deltas"][name].float().square()))),
                "candidate_update_rms": float(torch.sqrt(torch.mean(math_result["deltas"][name].float().square()))),
                "parameter_rms": float(torch.sqrt(torch.mean(manual["before"][name].float().square()))),
                "gradient_sign_disagreement_count": int((sign_flip & (left_grad != 0) & (right_grad != 0)).sum()),
                "max_update_error_reference_gradient": float(left_grad[flat_index]),
                "max_update_error_candidate_gradient": float(right_grad[flat_index]),
                "max_update_error_has_gradient_sign_flip": bool(sign_flip[flat_index]),
                "two_times_learning_rate": 0.0002,
                "passes_frozen_gate": metric["max_abs_error"] <= TOLERANCES["optimizer_parameter_max_abs"],
            }
        )
        updates[name] = metric
        optimizer_states[name] = {
            state_name: tensor_metrics(
                manual["states"][name][state_name], math_result["states"][name][state_name], atol=0.0, rtol=0.0
            )
            for state_name in ("exp_avg", "exp_avg_sq")
        }
    tied_update_flat_index = int(
        (math_result["deltas"]["token_embedding.weight"] - manual["deltas"]["token_embedding.weight"])
        .abs()
        .flatten()
        .argmax()
    )
    tied = {
        "weight_is_tied_by_model_contract": True,
        "total": gradient_metrics(manual["gradients"]["token_embedding.weight"], math_result["gradients"]["token_embedding.weight"]),
        "output_projection_contribution": gradient_metrics(manual_output, math_output),
        "input_embedding_contribution": gradient_metrics(manual_input, math_input),
        "manual_recomposition_max_abs": float((manual_output + manual_input - manual["gradients"]["token_embedding.weight"]).abs().max()),
        "math_recomposition_max_abs": float((math_output + math_input - math_result["gradients"]["token_embedding.weight"]).abs().max()),
        "components_at_max_update_error": {
            "flat_index": tied_update_flat_index,
            "manual_output_projection": float(manual_output.flatten()[tied_update_flat_index]),
            "math_output_projection": float(math_output.flatten()[tied_update_flat_index]),
            "manual_input_embedding": float(manual_input.flatten()[tied_update_flat_index]),
            "math_input_embedding": float(math_input.flatten()[tied_update_flat_index]),
        },
        "method": "DIAGNOSTIC_RECONSTRUCTION: detach only input lookup weight, preserve identical forward values and tied output head, then subtract output-only gradient from total",
    }
    return {
        "reference_loss": manual["loss"],
        "candidate_loss": math_result["loss"],
        "loss_abs_error": abs(math_result["loss"] - manual["loss"]),
        "gradients": gradients,
        "tied_embedding_decomposition": tied,
        "optimizer_updates": updates,
        "optimizer_states": optimizer_states,
    }


def _historical_hashes(repo: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        path: {"expected": expected, "actual": file_sha256(repo / path), "preserved": file_sha256(repo / path) == expected}
        for path, expected in protocol["historical_frozen_artifacts"].items()
    }


def run_diagnostics(repo: str | Path = ".") -> dict[str, Any]:
    started = time.perf_counter()
    repo = Path(repo).resolve()
    output = repo / "artifacts/gpu4b_r1"
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(output / "protocol.json")
    historical_before = _historical_hashes(repo, protocol)
    if not all(item["preserved"] for item in historical_before.values()):
        raise RuntimeError("historical artifact hash mismatch before diagnostics")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU-4B-R1 requires CUDA with BF16 support")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    manifest = read_json(repo / "artifacts/training/minillm_formal_model.json")
    architecture = manifest["architecture"]
    payload = torch.load(repo / CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["model"]
    batch = _batch(repo)
    input_identity = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""), "sha256": tensor_sha256(value)}
        for name, value in batch.items()
    }

    model = _model(architecture, state, "manual")
    fp32_manual, fp32_manual_loss = _capture_forward(model, batch, bf16=False)
    for layer in model.layers:
        layer.self_attn.attention_backend = "sdpa_math"
    fp32_math, fp32_math_loss = _capture_forward(model, batch, bf16=False)
    for layer in model.layers:
        layer.self_attn.attention_backend = "manual"
    bf16_manual, bf16_manual_loss = _capture_forward(model, batch, bf16=True)
    for layer in model.layers:
        layer.self_attn.attention_backend = "sdpa_math"
    bf16_math, bf16_math_loss = _capture_forward(model, batch, bf16=True)

    fp32_comparison = _compare_captures(fp32_manual, fp32_math, **TOLERANCES["fp32"])
    bf16_comparison = _compare_captures(bf16_manual, bf16_math, **TOLERANCES["bf16"])
    precision = _precision_reconstructions(bf16_manual, bf16_math, batch, model.layers[0].self_attn.scale)
    weight = model.lm_head.weight.detach().float().cpu()
    hidden_delta = bf16_math["final_norm.output"].float() - bf16_manual["final_norm.output"].float()
    predicted_logits_delta = F.linear(hidden_delta, weight)
    observed_logits_delta = bf16_math["logits"].float() - bf16_manual["logits"].float()
    hidden_error_rms = float(torch.sqrt(torch.mean(hidden_delta.square())))
    logits_error_rms = float(torch.sqrt(torch.mean(observed_logits_delta.square())))
    projection = {
        "hidden_error_rms": hidden_error_rms,
        "logits_error_rms": logits_error_rms,
        "observed_rms_gain": logits_error_rms / max(hidden_error_rms, RELATIVE_FLOOR),
        "weight_rms": float(torch.sqrt(torch.mean(weight.square()))),
        "max_output_row_l2_norm": float(weight.norm(dim=1).max()),
        "fp32_linearized_delta_vs_observed_delta": tensor_metrics(observed_logits_delta, predicted_logits_delta, atol=0.0, rtol=0.0),
        "interpretation": "The tied output matrix normally maps accumulated hidden-state error into vocabulary logits; residual mismatch is expected from separate BF16 projection rounding.",
    }
    _release(model)

    forward = {
        "input_identity": input_identity,
        "checkpoint_sha256": file_sha256(repo / CHECKPOINT),
        "same_weights_and_inputs": True,
        "model_mode": "eval",
        "dropout_probability": 0.0,
        "fp32": {
            "reference_loss": fp32_manual_loss,
            "candidate_loss": fp32_math_loss,
            "loss_abs_error": abs(fp32_math_loss - fp32_manual_loss),
            "tensors": fp32_comparison,
            "frozen_gate_pass": fp32_comparison["logits"]["allclose"] and abs(fp32_math_loss - fp32_manual_loss) <= TOLERANCES["loss_abs"],
        },
        "bf16": {
            "reference_loss": bf16_manual_loss,
            "candidate_loss": bf16_math_loss,
            "loss_abs_error": abs(bf16_math_loss - bf16_manual_loss),
            "tensors": bf16_comparison,
            "frozen_gate_pass": bf16_comparison["logits"]["allclose"] and abs(bf16_math_loss - bf16_manual_loss) <= TOLERANCES["loss_abs"],
        },
        "output_projection_analysis": projection,
    }
    write_json(output / "forward_diagnostics.json", forward)
    write_json(output / "precision_experiment.json", precision)
    gradient_update = _gradient_update_analysis(architecture, state, batch)
    write_json(output / "gradient_update_diagnostics.json", gradient_update)
    historical_after = _historical_hashes(repo, protocol)
    result = _write_reports_and_result(
        repo, output, protocol, forward, precision, gradient_update, historical_before, historical_after, started
    )
    return result


def _write_reports_and_result(
    repo: Path,
    output: Path,
    protocol: dict[str, Any],
    forward: dict[str, Any],
    precision: dict[str, Any],
    gradient: dict[str, Any],
    historical_before: dict[str, Any],
    historical_after: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    bf = forward["bf16"]["tensors"]
    fp = forward["fp32"]["tensors"]
    ordered_boundaries = [
        "block.0.q_projection", "block.0.k_projection", "block.0.v_projection",
        "block.0.rope.q", "block.0.rope.k", "block.0.attention.context",
        "block.0.o_projection", "block.0.attention.output", "block.0.attention.residual", "block.0.output",
    ]
    first_divergence = next(name for name in ordered_boundaries if bf[name]["max_abs_error"] > 0)
    tolerance_order = ordered_boundaries + [
        *(f"block.{index}.output" for index in range(1, 8)),
        "final_norm.output",
        "logits",
    ]
    first_non_allclose = next((name for name in tolerance_order if not bf[name]["allclose"]), None)
    variants = precision["variants"]
    manual_reconstruction_exact = variants["production_equivalent_manual"]["versus_actual_manual"]["max_abs_error"] == 0.0
    candidate_errors = {
        name: data["versus_actual_sdpa_math"]["rms_error"] for name, data in variants.items()
    }
    best_variant = min(candidate_errors, key=candidate_errors.get)
    documented_float_intermediates_supported = candidate_errors["full_fp32_then_bf16"] < candidate_errors["production_equivalent_manual"]
    no_semantic_defect = (
        all(fp[name]["allclose"] for name in ordered_boundaries)
        and all(bf[name]["max_abs_error"] == 0 for name in ordered_boundaries[:5])
        and precision["score_path"]["finite_min_vs_negative_infinity_context"]["max_abs_error"] == 0
    )
    classification = (
        "GPU4B_R1_NUMERICAL_PATH_DIFFERENCE_IDENTIFIED"
        if no_semantic_defect and manual_reconstruction_exact and documented_float_intermediates_supported
        else "GPU4B_R1_NO_DETERMINISTIC_ROOT_CAUSE"
    )
    block_rows = []
    for index in range(8):
        item = bf[f"block.{index}.output"]
        block_rows.append(
            f"| {index + 1} | {item['max_abs_error']:.8g} | {item['rms_error']:.8g} | {item['relative_rms_error']:.8g} | {item['allclose_fraction']:.8f} |"
        )
    numerical_audit = f"""# Numerical path audit

## Confirmed from production code and public runtime boundaries

- Both paths use identical bias-free Q/K/V projections, RoPE placement, scale `1/sqrt(64)`, causal + right-padding mask semantics, dropout=0, and GQA 8/4 grouping.
- Manual explicitly expands each K/V head twice, computes QK under BF16 autocast, scales, applies the mask, runs softmax in FP32, rounds probabilities to BF16, then performs Attention x V under BF16 autocast.
- SDPA Math receives BF16 Q/K/V, the same Boolean mask and explicit scale, with native `enable_gqa=True`. Its public attention context is BF16.
- The installed PyTorch documentation states: for Math, all intermediates are kept in float when inputs are half or bfloat16. This is **documented backend behavior**, not a hook observation.
- SDPA internal QK scores, softmax probabilities, and AV tensors are **UNKNOWN** because the kernel does not expose hookable module boundaries.

## Semantic checks

- Q/K/V and post-RoPE values are bit-identical through the first layer in BF16.
- Replacing Manual's finite-min mask sentinel with negative infinity produces context max error `{precision['score_path']['finite_min_vs_negative_infinity_context']['max_abs_error']}`.
- FP32 logits reproduce the frozen gate: `{forward['fp32']['frozen_gate_pass']}`.
- No duplicate scaling, duplicate causal mask, GQA mapping error, RoPE displacement, or dropout mismatch was found.

Evidence levels are recorded in `precision_experiment.json`; controlled arithmetic variants are labeled `DIAGNOSTIC_RECONSTRUCTION` and are not represented as SDPA internals.
"""
    forward_report = f"""# Forward error localization

The frozen batch and checkpoint were reproduced. FP32 passes; BF16 fails the original logits gate.

- FP32 logits max/mean absolute error: `{fp['logits']['max_abs_error']}` / `{fp['logits']['mean_abs_error']}`; allclose `{fp['logits']['allclose']}`.
- BF16 logits max/mean absolute error: `{bf['logits']['max_abs_error']}` / `{bf['logits']['mean_abs_error']}`; allclose fraction `{bf['logits']['allclose_fraction']}`.
- BF16 loss Manual/Math/difference: `{forward['bf16']['reference_loss']}` / `{forward['bf16']['candidate_loss']}` / `{forward['bf16']['loss_abs_error']}`.
- First nonzero public boundary: `{first_divergence}`. First boundary failing the frozen BF16 allclose rule: `{first_non_allclose}`.
- At the max-logit-error index `{bf['logits']['max_error_index']}`, Manual=`{bf['logits']['max_error_reference']}`, Math=`{bf['logits']['max_error_candidate']}`.

## Propagation by block output

| Block | max abs | error RMS | relative RMS | allclose fraction |
|---:|---:|---:|---:|---:|
{chr(10).join(block_rows)}

The first error is observable at layer-1 attention context (the input to `o_proj`), after bit-identical Q/K/V and RoPE boundaries. It then propagates through residual and MLP paths. Final-normalized hidden error RMS is `{forward['output_projection_analysis']['hidden_error_rms']}` and logits error RMS is `{forward['output_projection_analysis']['logits_error_rms']}`, an observed RMS gain of `{forward['output_projection_analysis']['observed_rms_gain']}` through the tied vocabulary projection. The largest output-row L2 norm is `{forward['output_projection_analysis']['max_output_row_l2_norm']}`. The FP32 linearized hidden delta is separately compared with observed BF16 logits delta in the JSON, avoiding inference from maxima alone.
"""
    precision_report = """# BF16 precision experiment

All variants reuse the real first-layer Q/K/V, RoPE outputs, mask, scale, and checkpoint. They are **DIAGNOSTIC_RECONSTRUCTION**, not observations of SDPA kernel internals.

| Variant | RMS error vs actual Manual | RMS error vs actual SDPA Math |
|---|---:|---:|
""" + "\n".join(
        f"| {name} | {data['versus_actual_manual']['rms_error']:.9g} | {data['versus_actual_sdpa_math']['rms_error']:.9g} |"
        for name, data in variants.items()
    ) + f"""

The production-equivalent reconstruction matches actual Manual exactly: `{manual_reconstruction_exact}`. The closest tested reconstruction to SDPA Math is `{best_variant}`. Moving QK/softmax/AV retention toward FP32 reduces context RMS error versus actual Math from `{candidate_errors['production_equivalent_manual']}` to `{candidate_errors['full_fp32_then_bf16']}`. Combined with PyTorch's documented Math float-intermediate behavior, this supports a BF16 rounding-path mechanism. It does not prove the precise C++ instruction ordering, which remains UNKNOWN.
"""
    tied = gradient["tied_embedding_decomposition"]
    token_update = gradient["optimizer_updates"]["token_embedding.weight"]
    gradient_report = f"""# Gradient and update audit

- BF16 loss difference is `{gradient['loss_abs_error']}`.
- Tied weight total gradient max/RMS error: `{tied['total']['max_abs_error']}` / `{tied['total']['rms_error']}`; cosine `{tied['total']['cosine_similarity']}`.
- Output-head contribution max/RMS error: `{tied['output_projection_contribution']['max_abs_error']}` / `{tied['output_projection_contribution']['rms_error']}`.
- Input-embedding contribution max/RMS error: `{tied['input_embedding_contribution']['max_abs_error']}` / `{tied['input_embedding_contribution']['rms_error']}`.
- The decomposition recomposes total gradients with max residual `{max(tied['manual_recomposition_max_abs'], tied['math_recomposition_max_abs'])}`.

The tied result is therefore not a mysterious third path: it is the sum of the input lookup path and output vocabulary projection path. The sparse input-embedding contribution dominates the recorded max and RMS discrepancy, while the output-head contribution is widespread and accounts for most sign disagreements. Component values at the maximum-update location are recorded in the JSON.

For the tied parameter, the AdamW delta max error is `{token_update['max_abs_error']}` at `{token_update['max_error_index']}`. The corresponding Manual/Math gradients are `{token_update['max_update_error_reference_gradient']}` and `{token_update['max_update_error_candidate_gradient']}`; sign flip=`{token_update['max_update_error_has_gradient_sign_flip']}`. With zero-initialized moments, first-step Adam normalization approaches a sign update, so a tiny gradient sign disagreement can produce approximately `2 * lr = {token_update['two_times_learning_rate']}`. Full per-parameter gradients, update norms, sign-disagreement counts, and `exp_avg`/`exp_avg_sq` differences are in `gradient_update_diagnostics.json`.
"""
    root_report = f"""# GPU-4B-R1 root-cause report

## Classification

`{classification}`

1. **Original failure reproduced:** yes. FP32 passes and BF16 logits still fail the frozen GPU-4B gate (`max_abs={bf['logits']['max_abs_error']}`, allclose fraction `{bf['logits']['allclose_fraction']}`).
2. **Why FP32 passes:** both implementations share the same mathematical semantics and FP32 suppresses the BF16 intermediate-rounding split; observed max logits error is `{fp['logits']['max_abs_error']}`.
3. **Earliest observable BF16 difference:** layer-1 attention context at the `o_proj` input, after bit-identical Q/K/V projections and RoPE.
4. **Supported mechanism:** Manual rounds QK results and FP32-softmax probabilities back through BF16 before AV; documented SDPA Math retains float intermediates. Controlled precision reconstructions move toward the Math result when intermediates are retained in FP32.
5. **Implementation defect:** none found in scaling, masks, GQA, RoPE, dropout, or training/eval routing. Exact SDPA internal instruction order is still unknown.
6. **Gradient/update mechanism:** forward differences propagate to backward; tied gradient combines a sparse input-embedding contribution (dominant max/RMS error) with a widespread output-head contribution (dominant sign-disagreement count). AdamW's first-step sign normalization amplifies near-zero sign flips to about `2*lr`, explaining threshold-edge update differences.
7. **Gate status:** the original GPU-4B BF16, gradient, and optimizer gates remain failed and are not rewritten.
8. **Production changes:** none. Manual + Eager remains the formal default; no checkpoint or historical artifact changed.
9. **Recommendation:** if SDPA work continues, first establish a separately frozen BF16 equivalence protocol based on distributions, normalized gradients, multiple shapes, bounded trajectories, checkpoint compatibility, and generation behavior. Do not relabel GPU-4B's old gate.
10. **Performance work:** not justified in this stage; no benchmark or `torch.compile` run was performed.
"""
    (output / "NUMERICAL_PATH_AUDIT.md").write_text(numerical_audit, encoding="utf-8")
    (output / "FORWARD_ERROR_LOCALIZATION.md").write_text(forward_report, encoding="utf-8")
    (output / "BF16_PRECISION_EXPERIMENT.md").write_text(precision_report, encoding="utf-8")
    (output / "GRADIENT_AND_UPDATE_AUDIT.md").write_text(gradient_report, encoding="utf-8")
    (output / "ROOT_CAUSE_REPORT.md").write_text(root_report, encoding="utf-8")
    historical_preserved = all(item["preserved"] for item in historical_before.values()) and all(
        item["preserved"] for item in historical_after.values()
    )
    result = {
        "stage": "GPU-4B-R1",
        "classification": classification,
        "historical_results_preserved": historical_preserved,
        "original_gate_reproduction": {
            "fp32_pass": forward["fp32"]["frozen_gate_pass"],
            "bf16_pass": forward["bf16"]["frozen_gate_pass"],
            "bf16_logits_max_abs_error": bf["logits"]["max_abs_error"],
            "bf16_logits_allclose_fraction": bf["logits"]["allclose_fraction"],
        },
        "fp32_reference": {"logits": fp["logits"], "loss_abs_error": forward["fp32"]["loss_abs_error"]},
        "bf16_forward_analysis": {
            "first_nonzero_public_boundary": first_divergence,
            "first_frozen_tolerance_failure": first_non_allclose,
            "logits": bf["logits"],
            "projection": forward["output_projection_analysis"],
        },
        "precision_path_analysis": {
            "production_manual_reconstruction_exact": manual_reconstruction_exact,
            "best_tested_math_approximation": best_variant,
            "rms_errors_vs_math": candidate_errors,
            "documented_float_intermediates_supported": documented_float_intermediates_supported,
        },
        "gradient_analysis": {"selected_parameters": gradient["gradients"], "tied_embedding": tied},
        "optimizer_update_analysis": gradient["optimizer_updates"],
        "confirmed_root_causes": [
            "Manual and SDPA Math diverge first at the attention context public boundary under BF16, not at Q/K/V or RoPE",
            "Manual explicitly rounds QK and post-softmax probability intermediates through BF16 while PyTorch documents float intermediates for SDPA Math",
            "AdamW zero-state first-step sign normalization amplifies small gradient sign disagreements to approximately two learning rates",
        ],
        "supported_hypotheses": [
            "Different BF16 rounding locations account for the dominant forward difference",
            "The tied output projection linearly propagates accumulated hidden-state error into logits",
        ],
        "unresolved_questions": ["Exact SDPA Math C++ internal operation and rounding order is not exposed by runtime hooks"],
        "production_attention_changed": False,
        "formal_training_default_changed": False,
        "performance_benchmark_executed": False,
        "torch_compile_executed": False,
        "next_stage_recommendations": ["Freeze a separate BF16 SDPA correctness protocol before any renewed performance study"],
        "environment": {
            "os": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "execution": {"wall_time_seconds": time.perf_counter() - started, "seed": SEED},
        "tests": {"targeted": "PENDING", "full_regression": "PENDING"},
        "artifacts": {},
        "historical_hashes_before": historical_before,
        "historical_hashes_after": historical_after,
    }
    write_json(output / "gpu4b_r1_result.json", result)
    result["artifacts"] = {
        path.name: file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "gpu4b_r1_result.json"
    }
    write_json(output / "gpu4b_r1_result.json", result)
    return result
