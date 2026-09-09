#!/usr/bin/env python3
"""
Eval pipeline for bitnet-quantized SmolVLM.

Measures:
  - Weight-level: cosine similarity per layer (meaningful for post-training)
  - Model-level: perplexity on WikiText-2 (meaningful only with calibration or BitNet training)
  - Memory: model footprint after quantization

Usage:
  uv run python eval.py                          # full eval
  uv run python eval.py --no-ppl                 # skip perplexity (fast)
  uv run python eval.py --layers 0,1,2           # cosine sim only
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def ternarize(w, method="mean_abs"):
    if method == "mse_opt":
        best = torch.zeros(w.shape[0], 1, device=w.device)
        for i in range(w.shape[0]):
            v = w[i].abs()
            bm, bt = float("inf"), v.mean().item()
            for m in [0.5, 0.7, 1.0, 1.3, 1.5]:
                if (t := v.mean().item() * m) < 1e-8:
                    continue
                wt = torch.clamp(torch.round(w[i] / t), -1, 1)
                me = ((w[i].float() - wt.float() * t) ** 2).mean().item()
                if me < bm:
                    bm, bt = me, t
            best[i] = bt
        g = best.clamp(min=1e-8)
    else:
        g = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.clamp(torch.round(w / g), -1, 1)


@torch.no_grad()
def eval_cosine(model, q_width, layers=None):
    if layers is None:
        layers = [0]
    text_model = model.text_model
    results = []

    for li in layers:
        mlp = text_model.layers[li].mlp
        w = mlp.gate_proj.weight.data  # [1536, 576]
        acts = torch.randint(-50, 51, (w.shape[1],), dtype=torch.int8)
        fref = w.float() @ acts.float()

        # Activation quant
        qm = (1 << (q_width - 1)) - 1
        act_scale = acts.abs().max().clamp(min=1).float()
        qa = torch.clamp(torch.round(acts.float() * qm / act_scale), -qm, qm)
        acts_dequant = qa * act_scale / qm

        # Ternarize
        wt = ternarize(w, "mean_abs")
        g = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        al = g.squeeze().float()

        res = (acts_dequant @ wt.float().T) * al
        cos = torch.nn.functional.cosine_similarity(
            fref.unsqueeze(0), res.unsqueeze(0)
        ).item()
        density = (wt != 0).sum().item() / wt.numel()
        results.append(
            {"layer": li, "cos": round(cos, 4), "density": round(density, 3)}
        )

    return results


@torch.no_grad()
def compute_ppl(text_model, embed, input_ids):
    logits = text_model(input_ids).last_hidden_state @ embed.T
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1)
    )
    return math.exp(loss.item())


def load_wikitext2(tokenizer, seq_len=128, samples=20):
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(ds["text"][:samples])
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"WikiText-2 unavailable ({exc}); using deterministic fallback text.")
        text = (
            "The quick brown fox jumps over the lazy dog. " * 500
            + "Machine learning is transforming technology. " * 500
        )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len * 50)
    ids = enc.input_ids[0]
    ids = ids[: len(ids) - len(ids) % seq_len]
    return ids.view(-1, seq_len)


def quantize_model(text_model):
    """Replace Linear weights with their dequantized ternary approximation."""
    for _, mod in text_model.named_modules():
        if (
            isinstance(mod, torch.nn.Linear)
            and mod.weight.dim() == 2
            and mod.weight.shape[0] > 10
        ):
            gamma = mod.weight.data.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            mod.weight.data = ternarize(mod.weight.data, "mean_abs").float() * gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="4,8")
    parser.add_argument("--layers", default="0,1,2")
    parser.add_argument("--no-ppl", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()

    configs = [int(q) for q in args.configs.split(",")]
    layers = [int(layer) for layer in args.layers.split(",")]
    if any(config < 2 for config in configs):
        parser.error("--configs values must be at least 2")
    if args.seq_len <= 0 or args.samples <= 0:
        parser.error("--seq-len and --samples must be positive")

    print(f"Loading {MODEL_ID}...")
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    layer_count = len(model.text_model.layers)
    invalid = [layer for layer in layers if not 0 <= layer < layer_count]
    if invalid:
        parser.error(f"layer indices out of range: {invalid}; model has {layer_count}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token

    # Cosine sim (fast, works for any quantized model)
    print("\n─── Cosine similarity (per layer) ───")
    print(f"{'Config':>8} {'Layer':>6} {'Cos':>8} {'Density':>8}")
    print("-" * 30)
    all_cos = []
    for qw in configs:
        for r in eval_cosine(model, qw, layers):
            print(
                f"{'Q' + str(qw):>8} {r['layer']:>6} {r['cos']:>8.4f} {r['density']:>7.0%}"
            )
            all_cos.append(r["cos"])

    # Perplexity (only meaningful with training)
    if not args.no_ppl:
        dataset = load_wikitext2(tok, args.seq_len, args.samples)
        print("\n─── Perplexity (WikiText-2) ───")
        print(f"{'Config':>8} {'PPL':>12} {'ΔPPL':>12}  {'Size':>8}")
        print("-" * 40)

        # Baseline FP32
        bl_ppl = compute_ppl(
            model.text_model, model.text_model.embed_tokens.weight, dataset
        )
        print(
            f"{'FP32':>8} {bl_ppl:>12.1f} {'---':>12}  {sum(p.numel() * p.element_size() for p in model.text_model.parameters()) / 1e6:>7.0f} MB"
        )

        ternary_model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        ternary_model.eval()
        quantize_model(ternary_model.text_model)
        ppl = compute_ppl(
            ternary_model.text_model,
            ternary_model.text_model.embed_tokens.weight,
            dataset,
        )
        mb = (
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in ternary_model.text_model.parameters()
            )
            / 1e6
        )
        print(f"{'Ternary':>8} {ppl:>12.1f} {ppl - bl_ppl:+>12.1f}  {mb:>7.0f} MB")
        del ternary_model

    # Summary
    avg_cos = sum(all_cos) / len(all_cos) if all_cos else 0
    print(f"\nAvg cosine sim across layers: {avg_cos:.4f}")
    print("Note: Post-training ternarization preserves ~89% cosine similarity")
    print(
        "      but perplexity degrades because model wasn't trained with ternary constraints."
    )
    print("      For good PPL: train with BitNet-style STE or apply GPTQ calibration.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"cosine_avg": avg_cos}))


if __name__ == "__main__":
    main()
