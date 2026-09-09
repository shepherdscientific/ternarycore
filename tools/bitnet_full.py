#!/usr/bin/env python3
"""
Full SmolVLM BitNet: ternarize vision encoder + connector + text decoder + INT4 embedding.

Converts all layers to low-precision:
  - Linear: ternary {-1,0,+1} with per-channel alpha (BitNet b1.58)
  - Conv2d:  ternary {-1,0,+1} with per-channel alpha (for vision patch embed)
  - Embedding: INT4 symmetric (range [-7,+7]) or FP4 E2M1
  - LayerNorm/Bias: FP32 (negligible params)

Usage:
  uv run python tools/bitnet_full.py                    # convert and profile
  uv run python tools/bitnet_full.py --qfmt fp4          # use FP4 embeddings
  uv run python tools/bitnet_full.py --model MODEL_ID    # convert another model
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# ── BitNet modules ───────────────────────────────────────────────


class BitNetLinear(nn.Module):
    """Linear with ternary weights + STE."""

    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.gamma = nn.Parameter(torch.ones(out_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.gamma.data = self.weight.abs().mean(dim=1).clamp(min=1e-8)

    def ternarize(self):
        g = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        wt = torch.clamp(torch.round(self.weight / g), -1, 1)
        return self.weight + (wt - self.weight).detach()

    def forward(self, x):
        wt = self.ternarize()
        alpha = self.gamma.unsqueeze(1) * math.sqrt(self.weight.shape[1])
        return F.linear(x, wt * alpha, self.bias)


class BitNetConv2d(nn.Module):
    """Conv2d with ternary weights + STE."""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(
            torch.empty(out_ch, in_ch // groups, *self.kernel_size)
        )
        self.gamma = nn.Parameter(torch.ones(out_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.gamma.data = (
            self.weight.abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp(min=1e-8)
            .squeeze()
        )

    def ternarize(self):
        g = self.weight.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        wt = torch.clamp(torch.round(self.weight / g), -1, 1)
        return self.weight + (wt - self.weight).detach()

    def forward(self, x):
        wt = self.ternarize()
        alpha = self.gamma.view(-1, 1, 1, 1) * math.sqrt(
            self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        )
        padding = self.padding
        if self.padding_mode != "zeros":
            if isinstance(self.padding, tuple):
                pad_h, pad_w = self.padding
            else:
                pad_h = pad_w = self.padding
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
            padding = 0
        return F.conv2d(
            x, wt * alpha, self.bias, self.stride, padding, self.dilation, self.groups
        )


def fp8_encode(val):
    """Encode float to FP8 E4M3 (1s,4e,bias7,3m). Range: ±448, precision: ~6%."""
    if abs(val) < 1e-10:
        return 0
    sgn = 1 if val < 0 else 0
    v = abs(val)
    e = min(max(math.floor(math.log2(v)) + 7, 0), 14)  # clamp exp
    m = (
        round((v / 2 ** (e - 7) - 1) * 8)
        if e > 0
        else round(v / 2 ** (-6) * 8)
    )
    m = max(0, min(m, 7))
    return (sgn << 7) | (e << 3) | m


def fp8_decode(encoded):
    """Decode FP8 E4M3 back to float."""
    if encoded == 0:
        return 0.0
    sgn = -1 if encoded >> 7 else 1
    e = (encoded >> 3) & 0xF
    m = encoded & 0x7
    if e == 0:  # subnormal
        return sgn * (m / 8) * 2 ** (-6)
    return sgn * (1 + m / 8) * 2 ** (e - 7)


class BitNetLayerNorm(nn.Module):
    """LayerNorm with FP8 E4M3 quantization via STE.
    Stores FP32 latents. Forward: quantizes to FP8, decodes back to FP32.
    Gradient passes through (STE): no quantization in backward."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))

    def _fp8_quant(self, x):
        """FP8 E4M3 quantization with STE: q = decode(encode(x)) with gradient passthrough."""
        q = torch.zeros_like(x)
        flat = x.flatten()
        for i in range(len(flat)):
            q_i = fp8_decode(fp8_encode(float(flat[i])))
            q.flatten()[i] = q_i
        return x + (q - x).detach()

    def forward(self, x):
        if self.elementwise_affine:
            w = self._fp8_quant(self.weight)
            b = self._fp8_quant(self.bias)
        else:
            w, b = None, None
        return F.layer_norm(x, self.normalized_shape, w, b, self.eps)


class BitNetEmbedding(nn.Module):
    """Embedding with INT4 or FP4 quantization.

    INT4: symmetric, range [-7, +7], scale = max(|w|) / 7 per row
    FP4:  E2M1 format (1s, 2e bias=1, 1m), range [-6, +6]
    """

    def __init__(self, num_embeddings, embedding_dim, qfmt="int4"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.qfmt = qfmt
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

    def quantize_int4(self):
        scale = self.weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 7
        q = torch.clamp(torch.round(self.weight / scale), -7, 7)
        return self.weight + (q * scale - self.weight).detach()

    def quantize_fp4(self, w=None):
        """FP4 E2M1: sign(1), exp(2,bias=1), mant(1) → value = (-1)^s * 2^(e-1) * (1 + m/2)"""
        if w is None:
            w = self.weight
        sgn = w.sign()
        aw = w.abs() + 1e-10
        # Manual FP4 encode: find closest FP4 value
        fp4_vals = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        fp4_vals = fp4_vals.to(aw.device)
        # Find closest for each element
        q = torch.zeros_like(aw)
        for fv in fp4_vals[fp4_vals >= 0]:
            mask = (aw - fv).abs() < (aw - q).abs()
            q[mask] = fv
        q = q * sgn
        return self.weight + (q - self.weight).detach()

    def forward(self, x):
        if self.qfmt == "int4":
            wq = self.quantize_int4()
        else:
            wq = self.quantize_fp4()
        return F.embedding(x, wq)


# ── Model conversion ─────────────────────────────────────────────


def convert_module(module, qfmt="int4", prefix=""):
    """Recursively convert nn.Linear, nn.Conv2d, nn.Embedding to BitNet versions."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        new_child = None

        if isinstance(child, nn.Linear):
            bn = BitNetLinear(
                child.in_features, child.out_features, child.bias is not None
            )
            with torch.no_grad():
                bn.weight.copy_(child.weight)
                if child.bias is not None:
                    bn.bias.copy_(child.bias)
                # gamma must be derived from the copied pretrained weight, not
                # the init-time (random) weight it was computed from.
                bn.gamma.data = bn.weight.abs().mean(dim=1).clamp(min=1e-8)
            new_child = bn

        elif isinstance(child, nn.Conv2d):
            bn = BitNetConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None,
                child.padding_mode,
            )
            with torch.no_grad():
                bn.weight.copy_(child.weight)
                if child.bias is not None:
                    bn.bias.copy_(child.bias)
                # gamma must be derived from the copied pretrained weight, not
                # the init-time (random) weight it was computed from.
                bn.gamma.data = (
                    bn.weight.abs()
                    .mean(dim=(1, 2, 3), keepdim=True)
                    .clamp(min=1e-8)
                    .squeeze()
                )
            new_child = bn

        elif isinstance(child, nn.Embedding):
            be = BitNetEmbedding(child.num_embeddings, child.embedding_dim, qfmt)
            with torch.no_grad():
                be.weight.copy_(child.weight)
            new_child = be

        elif isinstance(child, nn.LayerNorm):
            new_child = BitNetLayerNorm(
                child.normalized_shape, child.eps, child.elementwise_affine
            )
            if child.elementwise_affine:
                with torch.no_grad():
                    new_child.weight.copy_(child.weight)
                    new_child.bias.copy_(child.bias)

        if new_child is not None:
            setattr(module, name, new_child)
            print(
                f"  Converted {full_name}: {type(child).__name__} → {type(new_child).__name__}"
            )
        else:
            convert_module(child, qfmt, full_name)

    return module


def build_full_bitnet_vlm(qfmt="int4", model_id=None):
    """Load SmolVLM and convert all components to BitNet."""
    model_id = model_id or MODEL_ID
    print(f"Loading {model_id}...")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    print("\nConverting vision encoder...")
    convert_module(model.vision_model, qfmt, "vision")

    print("\nConverting connector...")
    convert_module(model.connector, qfmt, "connector")

    print("\nConverting text decoder...")
    convert_module(model.text_model, qfmt, "text")

    # Total params
    def count(prefix, mod):
        total = sum(p.numel() for p in mod.parameters())
        print(f"  {prefix}: {total / 1e6:.1f}M params")
        return total

    print("\n─── Parameter summary ───")
    total = 0
    for name, mod in [
        ("vision", model.vision_model),
        ("connector", model.connector),
        ("text", model.text_model),
    ]:
        total += count(name, mod)
    print(f"  Total: {total / 1e6:.1f}M")

    # Memory estimate (corrected)
    def _bits(mod, cls, b):
        return (
            sum(
                p.numel() * b
                for _, m in mod.named_modules()
                if isinstance(m, cls)
                for p in m.parameters()
            )
            // 8
        )

    vis_b = _bits(model.vision_model, (BitNetLinear, BitNetConv2d), 2)
    vis_ln = _bits(model.vision_model, BitNetLayerNorm, 8)
    conn_b = _bits(model.connector, BitNetLinear, 2)
    txt_b = _bits(model.text_model, BitNetLinear, 2)
    txt_e = _bits(model, BitNetEmbedding, 4)
    txt_ln = _bits(model.text_model, BitNetLayerNorm, 8)
    total_mb = (vis_b + vis_ln + conn_b + txt_b + txt_e + txt_ln) / 1024 / 1024
    fp32_mb = sum(p.numel() * 32 for p in model.parameters()) // 8 / 1024 / 1024
    print(
        f"  Memory: {total_mb:.0f}MB (ternary+INT4+FP8) vs {fp32_mb:.0f}MB FP32 = {fp32_mb / total_mb:.0f}x"
    )
    print(
        f"    Vision:  {vis_b / 1024 / 1024:.0f}MB (ternary) + {vis_ln / 1024 / 1024:.1f}MB (FP8 LN)"
    )
    print(f"    Connector: {conn_b / 1024 / 1024:.0f}MB (ternary)")
    print(
        f"    Text:   {txt_b / 1024 / 1024:.0f}MB (ternary) + {txt_e / 1024 / 1024:.0f}MB (INT4 embed) + {txt_ln / 1024 / 1024:.1f}MB (FP8 LN)"
    )

    return model


# ── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=MODEL_ID, help="HuggingFace model ID (default: %(default)s)"
    )
    parser.add_argument("--qfmt", default="int4", choices=["int4", "fp4"])
    args = parser.parse_args()

    model = build_full_bitnet_vlm(args.qfmt, args.model)

    # Quick sanity check: verify everything is BitNet
    n_bitnet = sum(
        1
        for _, m in model.named_modules()
        if isinstance(m, (BitNetLinear, BitNetConv2d, BitNetEmbedding))
    )
    print(f"\nBitNet modules: {n_bitnet}")
    print("Ready for evaluation with tools/eval_vlm.py --load-bitnet")


if __name__ == "__main__":
    main()
