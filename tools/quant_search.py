#!/usr/bin/env python3
"""Search quantization configurations for one representative model layer.

The reported storage metric amortizes ternary weights, one activation vector,
and scale metadata over the selected matrix. Accuracy is cosine similarity
against the floating-point matrix-vector product after quantize/dequantize.
"""

import itertools
import math

import torch
from transformers import AutoModel

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
LAYER_IDX = 0

SEARCH_SPACE = {
    "Q_WIDTH": [2, 3, 4, 8],
    "act_method": ["absmax", "p99", "p95"],
    "weight_method": ["mean_abs", "median_abs", "mse_opt"],
    "scale_fmt": ["Q15", "FP8_E5M2", "INT8"],
    "scale_gran": ["per_channel", "per_tensor"],
}


def quantize_acts(
    acts: torch.Tensor, q_width: int, method: str = "absmax"
) -> torch.Tensor:
    """Quantize and dequantize one activation vector."""
    if q_width < 2:
        raise ValueError("q_width must be at least 2")
    q_max = (1 << (q_width - 1)) - 1
    if method == "absmax":
        scale = acts.abs().max().clamp(min=1).float()
    elif method.startswith("p"):
        percentile = int(method[1:])
        scale = acts.abs().float().quantile(percentile / 100.0).clamp(min=1)
    else:
        raise ValueError(f"unknown activation method: {method}")
    quantized = torch.clamp(torch.round(acts.float() * q_max / scale), -q_max, q_max)
    return quantized * scale / q_max


def ternarize_weights(w: torch.Tensor, method: str = "mean_abs"):
    """Return ternary weights and their reconstruction scale."""
    if method == "mean_abs":
        gamma = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    elif method == "median_abs":
        gamma = w.abs().median(dim=1, keepdim=True).values.clamp(min=1e-8)
    elif method == "mse_opt":
        flat = w.reshape(w.shape[0], -1)
        best_gamma = torch.zeros(w.shape[0], 1, dtype=torch.float32, device=w.device)
        for row in range(w.shape[0]):
            values = flat[row].abs()
            best_mse = float("inf")
            best_threshold = values.mean().item()
            for multiplier in (0.5, 0.7, 1.0, 1.3, 1.5):
                threshold = values.mean().item() * multiplier
                if threshold < 1e-8:
                    continue
                ternary = torch.clamp(torch.round(flat[row] / threshold), -1, 1)
                mse = (
                    ((flat[row].float() - ternary.float() * threshold) ** 2)
                    .mean()
                    .item()
                )
                if mse < best_mse:
                    best_mse = mse
                    best_threshold = threshold
            best_gamma[row] = best_threshold
        gamma = best_gamma.clamp(min=1e-8)
    else:
        raise ValueError(f"unknown weight method: {method}")

    ternary = torch.clamp(torch.round(w / gamma), -1, 1)
    return ternary, gamma.squeeze(1)


def encode_scale(alpha: float, fmt: str) -> float:
    """Round-trip one non-negative scale through a candidate format."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if fmt == "Q15":
        return max(0, min(65535, round(alpha * 32768))) / 32768.0
    if fmt == "INT8":
        return max(0, min(127, round(alpha * 127))) / 127.0
    if fmt != "FP8_E5M2":
        raise ValueError(f"unknown scale format: {fmt}")
    if alpha < 2**-16:
        return 0.0

    exponent = math.floor(math.log2(alpha))
    mantissa = round((alpha / (2**exponent) - 1) * 4)
    if mantissa > 3:
        mantissa = 0
        exponent += 1
    if exponent > 15:
        exponent, mantissa = 15, 3
    if exponent < -14:
        # E5M2 subnormal step is 2^-16.
        mantissa = max(0, min(3, round(alpha / (2**-16))))
        return mantissa * (2**-16)
    return (1 + mantissa / 4) * (2**exponent)


def bits_per_param(
    q_width: int, scale_fmt: str, scale_gran: str, out_dim: int, in_dim: int
) -> float:
    """Calculate weight-equivalent storage bits per matrix parameter."""
    scale_bits = {"Q15": 16, "FP8_E5M2": 8, "INT8": 8}[scale_fmt]
    scale_overhead = (
        scale_bits / in_dim
        if scale_gran == "per_channel"
        else scale_bits / (out_dim * in_dim)
    )
    activation_overhead = q_width / out_dim
    return 2 + activation_overhead + scale_overhead


def evaluate_config(acts, weights, config: dict) -> dict:
    """Evaluate one quantization configuration."""
    out_dim, in_dim = weights.shape
    float_reference = weights.float() @ acts.float()
    acts_dequant = quantize_acts(acts, config["Q_WIDTH"], config["act_method"])
    ternary, alphas = ternarize_weights(weights, config["weight_method"])
    dot = acts_dequant @ ternary.float().T

    if config["scale_gran"] == "per_channel":
        decoded_alphas = torch.tensor(
            [encode_scale(alpha.item(), config["scale_fmt"]) for alpha in alphas],
            device=weights.device,
        )
    else:
        decoded_alpha = encode_scale(alphas.mean().item(), config["scale_fmt"])
        decoded_alphas = torch.full_like(alphas, decoded_alpha)

    result = dot * decoded_alphas
    cosine = torch.nn.functional.cosine_similarity(
        float_reference.unsqueeze(0), result.unsqueeze(0)
    ).item()
    bpp = bits_per_param(
        config["Q_WIDTH"], config["scale_fmt"], config["scale_gran"], out_dim, in_dim
    )
    return {"cos": cosine, "bpp": bpp, **config}


def pareto_frontier(results):
    """Find configs that minimize storage while maximizing cosine similarity."""
    frontier = []
    for result in results:
        dominated = any(
            other["bpp"] <= result["bpp"]
            and other["cos"] >= result["cos"]
            and (other["bpp"] < result["bpp"] or other["cos"] > result["cos"])
            for other in results
        )
        if not dominated:
            frontier.append(result)
    return sorted(frontier, key=lambda item: item["bpp"])


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    weights = model.text_model.layers[LAYER_IDX].mlp.gate_proj.weight.data
    torch.manual_seed(42)
    acts = torch.randint(-50, 51, (weights.shape[1],), dtype=torch.int8)

    keys = list(SEARCH_SPACE)
    values = list(SEARCH_SPACE.values())
    total = math.prod(len(value) for value in values)
    print(f"Search space: {total} configurations")

    results = []
    for index, combination in enumerate(itertools.product(*values), start=1):
        config = dict(zip(keys, combination))
        results.append(evaluate_config(acts, weights, config))
        if index % 100 == 0:
            print(f"  {index}/{total} evaluated...")

    frontier = pareto_frontier(results)
    print(f"\n─── Pareto frontier ({len(frontier)} configs) ───")
    print(
        f"{'BPP':>6} {'Cos':>6} {'Qw':>3} {'Act':>10} {'Wt':>12} "
        f"{'Scale':>12} {'S.gran':>12}"
    )
    print("-" * 70)
    for result in frontier:
        print(
            f"{result['bpp']:6.2f} {result['cos']:6.4f} "
            f"{result['Q_WIDTH']:3d} {result['act_method']:>10} "
            f"{result['weight_method']:>12} {result['scale_fmt']:>12} "
            f"{result['scale_gran']:>12}"
        )

    best_accuracy = max(frontier, key=lambda item: item["cos"])
    best_efficiency = max(frontier, key=lambda item: item["cos"] / item["bpp"])
    print(
        f"\nBest accuracy: Q{best_accuracy['Q_WIDTH']} "
        f"cos={best_accuracy['cos']:.4f} bpp={best_accuracy['bpp']:.2f}"
    )
    print(
        f"Best efficiency: Q{best_efficiency['Q_WIDTH']} "
        f"cos={best_efficiency['cos']:.4f} bpp={best_efficiency['bpp']:.2f}"
    )


if __name__ == "__main__":
    main()
