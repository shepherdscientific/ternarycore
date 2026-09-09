#!/usr/bin/env python3
"""
SmolVLM-256M → ternarycore weight converter.

Downloads SmolVLM-256M-Instruct, extracts linear layer weights, applies
BitNet b1.58 ternary quantization, and writes bit-exact pipeline test vectors.

Usage:
  python3 convert_smolvlm.py                          # download + analyze + vectors
  python3 convert_smolvlm.py --weights-only           # write weight statistics only
  python3 convert_smolvlm.py --no-download            # use cached model only
"""

import argparse
import json
import math
from pathlib import Path

import torch
from hardware_semantics import pack_ternary_weights
from torch import nn

# ── Config ────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
TERNARYCORE_COLS = 4  # parallel dot units
TERNARYCORE_DEPTH = 4  # vector length per dot

# ── BitNet ternary quantization ────────────────────────────────────


def ternarize_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    BitNet b1.58 ternary quantization:
      gamma = mean(|w|) per output channel
      w_hat = clamp(round(w / gamma), -1, +1)  (ste_round)
      alpha = gamma * sqrt(in_features)         # scale for matmul correctness

    Returns (w_ternary, alpha_scale)
    w_ternary: {-1, 0, +1} tensor
    alpha_scale: per-output-channel float scale
    """
    orig_shape = w.shape
    # Flatten to per-channel: weight is [out_features, in_features]
    if w.dim() == 2:
        out_dim, _ = w.shape
        flat = w
    elif w.dim() == 4:
        # Conv2D: [out, in, kH, kW] → view as [out, in*kH*kW]
        out_dim = w.shape[0]
        flat = w.view(out_dim, -1)
    else:
        return w, torch.ones(1)

    gamma = flat.abs().mean(dim=1, keepdim=True)  # per-channel
    gamma = gamma.clamp(min=1e-8)

    w_scaled = flat / gamma
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)

    alpha = gamma * math.sqrt(flat.shape[1])
    return w_ternary.reshape(orig_shape), alpha.squeeze()


def pack_ternary_chunk(weights: list[int]) -> int:
    """Pack one per-column ternary chunk using the shared RTL encoding."""
    return pack_ternary_weights([int(weight) for weight in weights], len(weights))


# ── Model loading ─────────────────────────────────────────────────


def load_smolvlm(model_id: str, device: str = "cpu", local_files_only=False):
    """Load a SmolVLM-compatible model for weight extraction."""
    from transformers import AutoModel

    print(f"Loading {model_id} on {device}...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=local_files_only,
    )
    model.eval()
    return model


# ── Weight extraction ─────────────────────────────────────────────


def extract_linear_weights(model, verbose: bool = True) -> list[dict]:
    """
    Walk the model, extract all nn.Linear weights.
    Returns list of {name, shape, weight, bias, alpha, w_ternary}.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.cpu()
            tern, alpha = ternarize_weights(w)

            layers.append(
                {
                    "name": name,
                    "shape": list(w.shape),
                    "weight": w,
                    "bias": module.bias.data.cpu() if module.bias is not None else None,
                    "w_ternary": tern,
                    "alpha": alpha,
                }
            )

            if verbose:
                n_tern = (tern != 0).sum().item()
                total = tern.numel()
                shape_str = str(list(w.shape))
                print(
                    f"  {name:60s} {shape_str:20s}  "
                    f"ternary={n_tern / total:.1%}  α={alpha[:3].tolist()}..."
                )

    return layers


# ── Test vector generation ────────────────────────────────────────


def generate_test_vectors(
    layers: list[dict], depth: int = 4, cols: int = 4, seed: int = 42
) -> list[dict]:
    """Generate deterministic, bit-exact ternary_pipeline test vectors."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not 1 <= cols <= 4:
        raise ValueError("cols must be between 1 and 4")
    generator = torch.Generator().manual_seed(seed)
    tests = []
    for layer in layers:
        w = layer["w_ternary"]
        if w.dim() != 2:
            continue
        out_dim, in_dim = w.shape

        # Take first depth x cols block of weights
        if in_dim < depth or out_dim < cols:
            continue

        w_block = w[:cols, :depth]  # [cols, depth]

        # Generate random int8 activations
        acts = torch.randint(
            -50, 51, (depth,), dtype=torch.int8, generator=generator
        ).tolist()

        absmax = max(abs(act) for act in acts) or 1
        inv = round((1 << 15) * 127 / absmax)
        quantized = [max(-127, min(127, (act * inv + (1 << 14)) >> 15)) for act in acts]
        alpha_q15 = [
            max(0, min(65535, round(float(value) * (1 << 15))))
            for value in layer["alpha"][:cols]
        ]

        # Compute expected: each col = sum(act * weight) over depth
        expected = []
        for c in range(cols):
            dot = sum(quantized[k] * int(w_block[c, k]) for k in range(depth))
            product = dot * alpha_q15[c]
            expected.append((product >> 15) + int((product & ((1 << 15) - 1)) != 0))

        # Pack weights into weight_enc format
        w_per_row = []
        for k in range(depth):
            col_weights = [int(w_block[c, k]) for c in range(cols)]
            w_per_row.append(pack_ternary_chunk(col_weights))

        tests.append(
            {
                "name": layer["name"],
                "activations": acts,
                "quantized": quantized,
                "inv": inv,
                "weights_packed": w_per_row,
                "expected": expected,
                "alphas_q15": alpha_q15,
            }
        )

    return tests


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SmolVLM → ternarycore converter")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Write weight statistics only; skip test-vector generation",
    )
    parser.add_argument(
        "--no-download", action="store_true", help="Use cached model, don't download"
    )
    parser.add_argument(
        "--model", default=MODEL_ID, help="HuggingFace model ID (default: %(default)s)"
    )
    parser.add_argument("--depth", type=int, default=TERNARYCORE_DEPTH)
    parser.add_argument("--cols", type=int, default=TERNARYCORE_COLS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("smolvlm_out"),
        help="Output directory",
    )
    args = parser.parse_args()

    if args.depth <= 0:
        parser.error("--depth must be positive")
    if not 1 <= args.cols <= 4:
        parser.error("--cols must be between 1 and 4")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    model = load_smolvlm(args.model, local_files_only=args.no_download)

    # Extract + ternary-quantize
    print("\nExtracting linear layers and applying BitNet ternary quantization...")
    layers = extract_linear_weights(model)

    # Dump weight stats
    stats = []
    for layer in layers:
        stats.append(
            {
                "name": layer["name"],
                "shape": layer["shape"],
                "ternary_density": float(
                    (layer["w_ternary"] != 0).sum() / layer["w_ternary"].numel()
                ),
                "alpha_min": float(layer["alpha"].min()),
                "alpha_max": float(layer["alpha"].max()),
                "has_bias": layer["bias"] is not None,
            }
        )

    stats_path = args.output_dir / "weight_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWeight stats → {stats_path}")

    if args.weights_only:
        return

    # Generate test vectors
    print("\nGenerating ternarycore test vectors...")
    tests = generate_test_vectors(
        layers, depth=args.depth, cols=args.cols, seed=args.seed
    )

    json_path = args.output_dir / "test_vectors.json"
    with open(json_path, "w") as f:
        json.dump(tests, f, indent=2, default=str)

    print(f"  Test vectors  → {json_path}  ({len(tests)} layers)")
    print("  Vectors are bit-exact for activation_quant → GEMM → scale.")
    print(f"\nTotal linear layers quantized: {len(layers)}")
    print(f"Total parameters: {sum(layer['w_ternary'].numel() for layer in layers):,}")
    print(
        "  non-zero ternary: "
        f"{sum((layer['w_ternary'] != 0).sum().item() for layer in layers):,}"
    )


if __name__ == "__main__":
    main()
