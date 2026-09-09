#!/usr/bin/env python3
"""
Benchmark any HuggingFace transformer model through the ternarycore pipeline.

Usage:
  python3 bench.py --model HuggingFaceTB/SmolVLM-256M-Instruct --qwidth 4
  python3 bench.py --model meta-llama/Llama-2-7b --qwidth 3 --method mse_opt --layers 0,2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel

# ── Quantizers ───────────────────────────────────────────────────


def quantize_acts(acts: torch.Tensor, q_width: int) -> torch.Tensor:
    """Quantize and dequantize one activation vector."""
    if q_width < 2:
        raise ValueError("q_width must be at least 2")
    qm = (1 << (q_width - 1)) - 1
    scale = acts.abs().max().clamp(min=1).float()
    quantized = torch.clamp(torch.round(acts.float() * qm / scale), -qm, qm)
    return quantized * scale / qm


def ternarize(w: torch.Tensor, method: str = "mean_abs"):
    if method == "mse_opt":
        best = torch.zeros(w.shape[0], 1)
        for i in range(w.shape[0]):
            v = w[i].abs()
            bm, bt = float("inf"), v.mean().item()
            for m in [0.5, 0.7, 1.0, 1.3, 1.5]:
                t = v.mean().item() * m
                if t < 1e-8:
                    continue
                wt = torch.clamp(torch.round(w[i] / t), -1, 1)
                me = ((w[i].float() - wt.float() * t) ** 2).mean().item()
                if me < bm:
                    bm, bt = me, t
            best[i] = bt
        g = best.clamp(min=1e-8)
    else:
        g = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.clamp(torch.round(w / g), -1, 1), g.squeeze()


# ── Model utilities ──────────────────────────────────────────────


def find_linear_layers(model, prefix=""):
    """Find all nn.Linear weight matrices in a model."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append(
                {
                    "name": f"{prefix}.{name}" if prefix else name,
                    "weight": module.weight.data,
                }
            )
    return layers


def get_mlp_weights(model):
    """Extract first MLP layer from common transformer architectures."""
    # Try common paths
    for suffix in [
        "text_model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.gate_proj",
        "transformer.h.0.mlp.c_fc",
        "layers.0.mlp.gate_proj",
        "encoder.layer.0.intermediate.dense",
    ]:
        parts = suffix.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            return obj.weight.data
        except AttributeError:
            continue
    # Fallback: find first large Linear
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] > 100:
            print(
                f"  Auto-detected MLP: {name} [{mod.weight.shape[0]}x{mod.weight.shape[1]}]"
            )
            return mod.weight.data
    return None


# ── Benchmark single config ──────────────────────────────────────


def bench_config(w, acts, q_width: int, method: str):
    """Run one quantization config on one weight matrix."""
    fref = w.float() @ acts.float()
    qa = quantize_acts(acts, q_width)
    wt, al = ternarize(w, method)
    res = (qa.float() @ wt.float().T) * al.float()
    cos = torch.nn.functional.cosine_similarity(
        fref.unsqueeze(0), res.unsqueeze(0)
    ).item()
    density = (wt != 0).sum().item() / wt.numel()
    return {"cos": round(cos, 4), "density": round(density, 3)}


# ── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Ternarycore benchmark")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--qwidth", default="4,8", help="Comma-separated Q_WIDTH values"
    )
    parser.add_argument(
        "--method",
        default="mse_opt",
        choices=["mean_abs", "mse_opt"],
        help="Weight ternarization method",
    )
    parser.add_argument(
        "--layers",
        default="0",
        help="Comma-separated indices into detected Linear layers",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Save results to JSON"
    )
    args = parser.parse_args()

    qwidths = [int(q) for q in args.qwidth.split(",")]
    layer_indices = [int(layer) for layer in args.layers.split(",")]
    if any(q_width < 2 for q_width in qwidths):
        parser.error("--qwidth values must be at least 2")

    print(f"Loading {args.model}...")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    linear_layers = find_linear_layers(model)
    if not linear_layers:
        print("Error: Could not find Linear layers", file=sys.stderr)
        return 1
    invalid = [index for index in layer_indices if not 0 <= index < len(linear_layers)]
    if invalid:
        parser.error(
            f"layer indices out of range: {invalid}; detected {len(linear_layers)} layers"
        )

    torch.manual_seed(args.seed)
    print(f"  Detected Linear layers: {len(linear_layers)}")
    print(f"  Q_WIDTHs: {qwidths}")
    print(f"  Method: {args.method}")
    print()

    all_results = []
    for layer_index in layer_indices:
        layer = linear_layers[layer_index]
        w = layer["weight"]
        acts = torch.randint(-50, 51, (w.shape[1],), dtype=torch.int8)
        print(f"Layer {layer_index}: {layer['name']} {list(w.shape)}")
        for qw in qwidths:
            r = bench_config(w, acts, qw, args.method)
            print(f"  Q{qw}: cos={r['cos']:.4f} density={r['density']:.0%}")
            all_results.append(
                {
                    "model": args.model,
                    "layer": layer_index,
                    "layer_name": layer["name"],
                    "q_width": qw,
                    **r,
                }
            )

    # Cross-model average
    avg = {}
    for qw in qwidths:
        vals = [r for r in all_results if r["q_width"] == qw]
        avg[f"Q{qw}"] = sum(v["cos"] for v in vals) / len(vals)
    print(f"\nAverages: {json.dumps(avg, indent=2)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
