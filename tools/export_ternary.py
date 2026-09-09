#!/usr/bin/env python3
"""
Export trained BitNet weights to ternarycore hardware format.

Generates:
  exported/weights.h    — C header with all packed weights + alphas
  exported/layer0_test.cpp — Verilator C++ testbench for layer 0
  exported/metadata.json   — shapes, alphas, layer info

Packing:
  Each ternary weight {-1, 0, +1} → 2-bit weight_enc: 00=0, 01=+1, 10=-1
  4 weights packed per byte: col0=bits[1:0], col1=bits[3:2], col2=bits[5:4], col3=bits[7:6]
  Per-channel alpha scales → Q15 unsigned (16-bit, 1.0 = 32768)

Usage:
  uv run python tools/export_ternary.py --checkpoint checkpoints/best.pt --output exported/
  uv run python tools/export_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --output exported/
  uv run python tools/export_ternary.py --synthetic --cols 4 --depth 4 --output /tmp/exported/
"""

import argparse
import json
import math
import os
import shlex
import sys
import textwrap
from pathlib import Path

import numpy as np
from hardware_semantics import pack_ternary_weights

# ── BitNet weight helpers (no torch dependency for synthetic mode) ───────────


def ternarize(weight_fp, gamma=None):
    """Quantize full-precision weight to {-1, 0, +1}.
    weight_fp: 2D numpy array (out_f, in_f)
    gamma:     1D numpy array (out_f,) — if None, computed as mean(abs)
    Returns:   ternary weight array same shape as weight_fp
    """
    if gamma is None:
        gamma = np.abs(weight_fp).mean(axis=1, keepdims=True)
        gamma = np.clip(gamma, 1e-8, None)
    else:
        gamma = gamma.reshape(-1, 1)
    wt = np.clip(np.round(weight_fp / gamma), -1, 1).astype(np.int8)
    return wt


def pack_weights(wt, cols=4):
    """Pack ternary weights into weight_enc format.
    wt:   2D array (out_f, in_f) of {-1, 0, +1}
    cols: number of output columns to pack per byte (default 4)
    Returns: 1D uint8 array of packed bytes
    Layout: for each depth position (in_f), for each group of COLS output channels:
      byte = enc(col0) | enc(col1)<<2 | enc(col2)<<4 | enc(col3)<<6
    """
    if wt.ndim != 2:
        raise ValueError(f"weights must be a 2D array, got shape {wt.shape}")
    if not 1 <= cols <= 4:
        raise ValueError("cols must be between 1 and 4 for byte-packed output")
    if not np.isin(wt, (-1, 0, 1)).all():
        raise ValueError("weights must contain only -1, 0, or +1")

    out_f, in_f = wt.shape
    if out_f % cols != 0:
        raise ValueError(f"out_f ({out_f}) must be divisible by cols ({cols})")
    groups = out_f // cols
    packed = np.zeros(in_f * groups, dtype=np.uint8)

    for d in range(in_f):
        for g in range(groups):
            row = [int(wt[g * cols + c, d]) for c in range(cols)]
            packed[d * groups + g] = pack_ternary_weights(row, cols)
    return packed


def quantize_alpha(alpha_float, precision=15):
    """Convert float alpha to Q15 unsigned fixed-point.
    alpha_float: per-channel alpha values (float array)
    Returns: uint16 array, 1.0 = 32768 = 2^precision

    Raises a warning if any alpha exceeds the Q15 range (2.0 = 65535),
    since the RTL's alpha port is only (PRECISION+1) bits wide and values
    > 65535 would silently wrap to 0 in the 16-bit field.
    """
    alpha_float = np.asarray(alpha_float)
    if not np.isfinite(alpha_float).all():
        raise ValueError("alpha values must all be finite")
    scale = 1 << precision
    max_q15 = (1 << (precision + 1)) - 1  # 65535 for precision=15
    raw = np.round(alpha_float * scale)
    if np.any(raw > max_q15):
        exceeded = raw[raw > max_q15]
        print(
            f"WARNING: {len(exceeded)} alpha channel(s) exceed Q15 max ({max_q15}, ~2.0): "
            f"max={raw.max():.0f} (would wrap to {raw.max() % (max_q15 + 1)} in the HW). "
            f"Clipping to {max_q15}."
        )
    if np.any(raw < 0):
        neg = raw[raw < 0]
        print(
            f"WARNING: {len(neg)} alpha channel(s) are negative (min={raw.min():.0f}). "
            f"Clipping to 0."
        )
    return np.clip(raw, 0, max_q15).astype(np.uint16)


def compute_gemm(wt, acts):
    """Software reference: for each output column, dot product with activations.
    wt:   2D (out_f, in_f) ternary weights {-1, 0, +1}
    acts: 1D (in_f,) int8 activations
    Returns: 1D (out_f,) int32 dot products
    """
    return np.dot(wt.astype(np.int32), acts.astype(np.int32))


def scale_results(dots, alphas, precision=15):
    """Apply Q15 per-channel scale: round(dot * alpha >> precision).
    dots:    1D (out_f,) int32
    alphas:  1D (out_f,) uint16 Q15
    Returns: 1D (out_f,) int32
    """
    scale = 1 << precision
    # Use int64 for intermediate product
    prod = dots.astype(np.int64) * alphas.astype(np.int64)
    shifted = prod >> precision
    # Round: check if any truncated bits are set
    trunc = prod & (scale - 1)
    round_up = (trunc != 0).astype(np.int64)
    return (shifted + round_up).astype(np.int32)


# ── Synthetic weight generation ─────────────────────────────────────────────


def gen_synthetic_weights(out_f, in_f, sparsity=0.5, seed=42):
    """Generate random ternary weights with controlled sparsity."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be between 0 and 1")
    nonzero_probability = 1.0 - sparsity
    rng = np.random.default_rng(seed)
    wt = rng.choice(
        [-1, 0, 1],
        size=(out_f, in_f),
        p=[nonzero_probability / 2, sparsity, nonzero_probability / 2],
    )
    alphas = rng.uniform(0.5, 2.0, size=out_f)
    return wt, alphas


# ── HuggingFace model loading ────────────────────────────────────────────────


def load_hf_model(model_id):
    """Load a HuggingFace model and convert to BitNet, returning extracted layers."""
    print(f"Loading {model_id}...", file=sys.stderr)
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    # If the model has text_model, use it; otherwise use the model directly
    core = getattr(model, "text_model", model)
    # Also try common container names
    for attr in ("model", "transformer", "backbone"):
        if hasattr(core, attr):
            core = getattr(core, attr)

    layers = _extract_bitnet_layers(core)
    if not layers:
        # Fallback: convert nn.Linear layers and extract
        layers = _convert_and_extract(core)
    return layers


def _extract_bitnet_layers(module, prefix="", layers=None):
    """Recursively find BitNetLinear-like modules (have .gamma attr)."""
    if layers is None:
        layers = {}
    # BitNetLinear from bitnet_full.py or train_bitnet.py
    cls_name = type(module).__name__
    if (cls_name == "BitNetLinear" or cls_name == "Linear") and (
        hasattr(module, "gamma") or cls_name == "BitNetLinear"
    ):
        name = prefix if prefix else "layer"
        layers[name] = {
            "weight": module.weight.detach().cpu().numpy(),
            "gamma": module.gamma.detach().cpu().numpy(),
            "in_f": module.in_features,
            "out_f": module.out_features,
            "has_bias": module.bias is not None,
        }
        return layers
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        _extract_bitnet_layers(child, full, layers)
    return layers


def _convert_and_extract(module, prefix="", layers=None):
    """Convert plain nn.Linear layers to BitNet form and extract them.

    Used for HuggingFace models that were not trained with BitNetLinear.
    There is no learned per-channel gamma, so it is estimated from the
    weight statistics: gamma = mean(|w|) per output channel (same default
    used by ternarize() when gamma is None). This makes the --model path
    functional instead of silently producing "No layers extracted".
    """
    if layers is None:
        layers = {}
    import torch

    if isinstance(module, torch.nn.Linear):
        w = module.weight.detach().cpu().numpy()
        gamma = np.clip(np.abs(w).mean(axis=1, keepdims=True), 1e-8, None)
        name = prefix if prefix else "layer"
        layers[name] = {
            "weight": w,
            "gamma": gamma,
            "in_f": module.in_features,
            "out_f": module.out_features,
            "has_bias": module.bias is not None,
        }
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        _convert_and_extract(child, full, layers)
    return layers


# ── Checkpoint loading ──────────────────────────────────────────────────────


def load_checkpoint(ckpt_path):
    """Load a training checkpoint and extract BitNet layers from state dict."""
    print(f"Loading checkpoint {ckpt_path}...", file=sys.stderr)
    import torch

    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Checkpoint could be a full state dict or a nested dict
    if isinstance(data, dict) and "model_state" in data:
        state = data["model_state"]
    else:
        state = data

    # Extract layer info by matching gamma/weight keys in state dict
    layers = {}
    weight_keys = {k for k in state if k.endswith(".weight")}
    for wk in sorted(weight_keys):
        # Find matching gamma key: same prefix
        prefix = wk[: -len(".weight")]
        gk = f"{prefix}.gamma"
        if gk in state and wk in state:
            w = state[wk].numpy()
            g = state[gk].numpy()
            # Infer in_f, out_f from shape
            if w.ndim == 2:
                out_f, in_f = w.shape
            elif w.ndim == 1:
                out_f, in_f = w.shape[0], 1
            else:
                continue
            layers[prefix] = {
                "weight": w,
                "gamma": g,
                "in_f": in_f,
                "out_f": out_f,
                "has_bias": f"{prefix}.bias" in state,
            }

    if not layers:
        raise ValueError(
            f"No BitNet layers found in checkpoint. "
            f"Expected keys with .weight and .gamma pairs. "
            f"Found weight keys: {sorted(weight_keys)[:10]}"
        )
    return layers


# ── C header generation ──────────────────────────────────────────────────────


def generate_header(layers, cols=4):
    """Generate C header with packed weights and alphas."""
    lines = [
        "#ifndef EXPORTED_WEIGHTS_H",
        "#define EXPORTED_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        "// Packed ternary weights in weight_enc format.",
        f"// Each uint8_t packs {cols} ternary weights:",
        "//   bits[1:0] = col0, bits[3:2] = col1, bits[5:4] = col2, bits[7:6] = col3",
        "//   Encoding: 00=0, 01=+1, 10=-1",
        f"// Packing factor: {cols} columns per byte",
        "//",
        "// Per-channel alpha scales are Q15 unsigned: 1.0 = 32768",
        "",
    ]

    for name, info in layers.items():
        wt = (
            info["wt_ternary"]
            if "wt_ternary" in info
            else ternarize(info["weight"], info["gamma"])
        )
        out_f, in_f = wt.shape
        groups = out_f // cols
        packed = info["packed"] if "packed" in info else pack_weights(wt, cols)
        alphas = (
            info["alpha_q15"] if "alpha_q15" in info else quantize_alpha(info["alpha"])
        )

        # Sanitize name for C identifier
        cname = _c_ident(name)

        # Header comment
        lines.append(f"// ── {name} ({out_f}x{in_f}) ───────────────────────────────")
        lines.append(f"#define {cname.upper()}_COLS {out_f}")
        lines.append(f"#define {cname.upper()}_DEPTH {in_f}")
        lines.append(f"#define {cname.upper()}_GROUPS {groups}")
        lines.append("")

        # Packed weights array
        lines.append(f"static const uint8_t {cname}_weights[{len(packed)}] = {{")
        for i in range(0, len(packed), 16):
            chunk = packed[i : i + 16]
            hex_vals = ", ".join(f"0x{b:02X}" for b in chunk)
            lines.append(f"    {hex_vals},")
        lines.append("};")
        lines.append("")

        # Alpha scales array
        lines.append(f"static const uint16_t {cname}_alphas[{len(alphas)}] = {{")
        for i in range(0, len(alphas), 8):
            chunk = alphas[i : i + 8]
            hex_vals = ", ".join(f"0x{a:04X}" for a in chunk)
            lines.append(f"    {hex_vals},")
        lines.append("};")
        lines.append("")

    lines.append("#endif  // EXPORTED_WEIGHTS_H")
    return "\n".join(lines)


# ── Verilator testbench generation ──────────────────────────────────────────


def generate_testbench(layers, name, cols=4, depth=4, seed=42):
    """Generate a Verilator C++ testbench for one layer.

    The testbench computes expected values ON THE FLY in C++ using the same
    arithmetic as the HW pipeline (activation_quant → GEMM → scale), so
    any rounding differences between Python and Verilator do not affect
    the pass/fail verdict.

    Uses the first COLS columns and first DEPTH elements of the named layer.
    """
    if name not in layers:
        raise KeyError(f"Layer '{name}' not found. Available: {list(layers.keys())}")

    info = layers[name]
    rng = np.random.default_rng(seed)
    wt = info["wt_ternary"]
    alphas = info["alpha_q15"]

    out_f, in_f = wt.shape
    if out_f < cols:
        raise ValueError(f"Layer {name} has only {out_f} cols, need at least {cols}")
    if in_f < depth:
        raise ValueError(f"Layer {name} has only {in_f} depth, need at least {depth}")

    # Take the first cols × depth block
    wt_block = wt[:cols, :depth]  # (cols, depth)
    alpha_block = alphas[:cols]  # (cols,)

    # Generate random int8 activations
    acts = rng.integers(-50, 51, size=depth, dtype=np.int8)

    # Compute inv for this test
    absmax = max(abs(int(a)) for a in acts) or 1
    q_max = 127
    inv = round((1 << 15) * q_max / absmax)  # Q15 inv = 2^15 * 127 / absmax

    # Pack weight_enc rows
    enc_map = {0: 0, 1: 1, -1: 2}
    weight_rows = []
    for d in range(depth):
        byte = 0
        for c in range(cols):
            w = int(wt_block[c, d])
            enc = enc_map.get(w, 0)
            byte |= enc << (2 * c)
        weight_rows.append(byte)

    # Build ternary weight matrix as C++ 2D array
    weight_init_lines = []
    for c in range(cols):
        row_vals = ", ".join(str(int(wt_block[c, d])) for d in range(depth))
        weight_init_lines.append(f"    {{{row_vals}}},")

    # Alpha occupies 16 bits per column and fits uint64_t for the supported
    # byte-packed 1..4-column formats.
    alpha_packed = 0
    for c in range(cols):
        alpha_packed |= int(alpha_block[c]) << (16 * c)

    if cols == 1:
        capture_result = "hw_result[0] = (int32_t)dut->result;"
    elif cols == 2:
        capture_result = """hw_result[0] = (int32_t)(dut->result & 0xFFFFFFFFULL);
            hw_result[1] = (int32_t)(dut->result >> 32);"""
    else:
        capture_result = f"""for (int c = 0; c < {cols}; c++)
                hw_result[c] = (int32_t)dut->result.at(c);"""

    # Build C++ source — includes inline SW reference computation
    acts_c = ", ".join(str(int(a)) for a in acts)
    wrows_c = ", ".join(str(w) for w in weight_rows)
    inv_c = str(inv)
    alpha_c = f"0x{alpha_packed:016X}ULL"

    source = f"""// Verilator testbench for {name} ({cols}x{depth} block)
// Auto-generated by tools/export_ternary.py
// HW pipeline: activation_quant → GEMM → per-channel scale
//
// Expected values are computed on the fly in C++ using the same arithmetic
// as the HW pipeline (int64_t, arithmetic right shift, Q15 rounding).

#include <verilated.h>
#include "Vternary_pipeline.h"
#include <cstdio>
#include <cstdint>
#include <cmath>

// ── Software reference: matches HW pipeline arithmetic ─────────────────────

// Activation quantizer (matches activation_quant.v)
static int8_t hw_quantize(int32_t x, int32_t inv) {{
    const int32_t round_amt = 1 << 14;  // 2^(PRECISION-1)
    const int32_t q_max = 127;
    const int32_t q_min = -127;
    int64_t product = (int64_t)x * (int64_t)inv;
    int64_t biased = product + round_amt;
    int64_t shifted = biased >> 15;  // arithmetic shift (PRECISION=15)
    if (shifted > q_max) return q_max;
    if (shifted < q_min) return q_min;
    return (int8_t)shifted;
}}

// Decode weight_enc byte to ternary value for one column
static int decode_weight(uint8_t w_enc, int col) {{
    int enc = (w_enc >> (2 * col)) & 3;
    if (enc == 0) return 0;
    if (enc == 1) return 1;
    return -1;  // enc == 2 (10)
}}

// Scale stage (matches ternary_scale.v)
static int32_t hw_scale(int32_t acc, uint16_t alpha_q15) {{
    const int precision = 15;
    int64_t prod = (int64_t)acc * (int64_t)alpha_q15;
    // Round: OR of all truncated bits
    int64_t trunc = prod & ((1LL << precision) - 1);
    int round_bit = (trunc != 0) ? 1 : 0;
    int64_t shifted = prod >> precision;
    return (int32_t)(shifted + round_bit);
}}


int main() {{
    Vternary_pipeline* dut = new Vternary_pipeline;
    int errors = 0;
    bool saw_valid = false;

    // ── Test vectors ─────────────────────────────────────────────
    // Ternary weight matrix: wt[col][depth] in {{-1, 0, +1}}
    int8_t wt[{cols}][{depth}] = {{
{chr(10).join(weight_init_lines)}
    }};
    int8_t acts[{depth}] = {{{acts_c}}};
    uint16_t alphas_q15[{cols}] = {{}};
    uint64_t alpha = {alpha_c};
    for (int c = 0; c < {cols}; c++)
        alphas_q15[c] = (uint16_t)((alpha >> (16 * c)) & 0xFFFF);

    int inv = {inv_c};

    // Packed weight rows for HW (one byte per depth position)
    int weight_rows[{depth}] = {{{wrows_c}}};

    // ── Compute SW reference ──────────────────────────────────────
    // 1. Quantize activations
    int8_t q[{depth}];
    for (int i = 0; i < {depth}; i++)
        q[i] = hw_quantize(acts[i], inv);

    // 2. GEMM: pipeline timing simulation
    //    activation_quant has 2-cycle latency (product reg, then q reg)
    //    weight_enc is delayed by the same 2 register stages
    //    Result: q[i] pairs with weight_rows[i]
    int32_t dots[{cols}] = {{0}};
    for (int i = 0; i < {depth}; i++) {{
        uint8_t w_enc = weight_rows[i];
        for (int c = 0; c < {cols}; c++)
            dots[c] += (int32_t)q[i] * (int32_t)decode_weight(w_enc, c);
    }}

    // 3. Scale: apply Q15 per-channel alpha
    int32_t expected[{cols}];
    for (int c = 0; c < {cols}; c++)
        expected[c] = hw_scale(dots[c], alphas_q15[c]);

    // ── Drive HW pipeline ─────────────────────────────────────────
    dut->rst_n = 0;
    for (int i = 0; i < 4; i++) {{ dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }}
    dut->rst_n = 1;

    for (int i = 0; i < {depth}; i++) {{
        dut->valid_in = 1;
        dut->activation = acts[i];
        dut->inv = inv;
        dut->weight_enc = weight_rows[i];
        dut->alpha = alpha;
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }}

    dut->valid_in = 0;
    int32_t hw_result[{cols}] = {{0}};
    for (int i = 0; i < {depth + 8}; i++) {{
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
        if (dut->valid_out) {{
            saw_valid = true;
            {capture_result}
        }}
    }}

    // ── Compare ───────────────────────────────────────────────────
    if (!saw_valid) {{
        printf("FAIL: valid_out was never asserted\\n");
        errors++;
    }}
    printf("{name} {cols}x{depth}:\\n");
    for (int c = 0; c < {cols}; c++) {{
        const char* status = (hw_result[c] == expected[c]) ? "OK" : "MIS";
        printf("  col%d: HW=%d SW=%d %s\\n", c, hw_result[c], expected[c], status);
        if (hw_result[c] != expected[c]) errors++;
    }}
    printf("--- %d error(s) ---\\n", errors);
    delete dut;
    return errors ? 1 : 0;
}}
"""
    return source


# ── Verilator build command ─────────────────────────────────────────────────


def build_command(test_dir, test_name="layer0_test", cols=4, depth=4):
    """Return a shell command to build the Verilator test."""
    rtl_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rtl"
    )
    rtl = shlex.quote(rtl_dir)
    test_cpp = shlex.quote(os.path.join(test_dir, f"{test_name}.cpp"))
    output = shlex.quote(os.path.join(test_dir, test_name))
    return textwrap.dedent(f"""\
    verilator --cc --build --exe --top-module ternary_pipeline \
        -Wno-BLKSEQ -Wno-UNUSEDPARAM -Wno-UNUSEDSIGNAL -Wno-PINMISSING \
        -Wno-EOFNEWLINE -Wno-WIDTHTRUNC -Wno-UNOPTFLAT -Wno-WIDTHEXPAND \
        -GVECTOR_LEN={depth} -GCOLS={cols} -GDATA_WIDTH=8 -GACC_WIDTH=32 \
        {rtl}/ternary_pipeline.v {rtl}/activation_quant.v {rtl}/ternary_weight.v \
        {rtl}/ternary_gemm.v {rtl}/ternary_dot.v {rtl}/ternary_scale.v \\
        {test_cpp} -o {output}
    """)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _c_ident(name):
    """Convert dotted name to valid C identifier."""
    safe = name.replace(".", "_").replace("-", "_").replace("/", "_")
    if not safe:
        return "layer"
    if safe[0].isdigit():
        safe = "l_" + safe
    return safe


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export BitNet weights to ternarycore hardware format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --checkpoint checkpoints/best.pt --output exported/
              %(prog)s --model HuggingFaceTB/SmolVLM-256M-Instruct --output exported/
              %(prog)s --synthetic --cols 4 --depth 4 --output /tmp/exported/
        """),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Path to trained checkpoint .pt file")
    src.add_argument("--model", help="HuggingFace model ID to load and convert")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic weights for testing",
    )
    parser.add_argument(
        "--output", default="exported", help="Output directory (default: exported/)"
    )
    parser.add_argument(
        "--cols", type=int, default=4, help="Pipeline columns (default: 4)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Pipeline depth / vector length (default: 4)",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Specific layer for testbench (default: first available)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Weight sparsity for synthetic mode (default: 0.5)",
    )
    args = parser.parse_args()

    if not 1 <= args.cols <= 4:
        parser.error("--cols must be between 1 and 4 for byte-packed output")
    if args.depth <= 0:
        parser.error("--depth must be positive")
    if not 0.0 <= args.sparsity <= 1.0:
        parser.error("--sparsity must be between 0 and 1")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = args.cols
    depth = args.depth
    seed = args.seed

    layers = {}  # name → {weight, gamma, in_f, out_f, has_bias, wt_ternary, alpha, alpha_q15, packed}

    # ── Load or generate weights ──────────────────────────────────
    if args.synthetic:
        print(
            f"Generating synthetic weights: out={cols}, in={depth}, sparsity={args.sparsity}",
            file=sys.stderr,
        )
        out_f, in_f = cols, depth
        wt_synth, alpha_synth = gen_synthetic_weights(out_f, in_f, args.sparsity, seed)
        gamma_synth = alpha_synth / math.sqrt(in_f)  # reverse engineer gamma from alpha
        layers["synthetic"] = {
            "weight": wt_synth.astype(np.float32),
            "gamma": gamma_synth.astype(np.float32),
            "in_f": in_f,
            "out_f": out_f,
            "has_bias": False,
            "wt_ternary": wt_synth,
            "alpha": alpha_synth,
        }

    elif args.checkpoint:
        layers = load_checkpoint(args.checkpoint)
        for name, info in layers.items():
            info["wt_ternary"] = ternarize(info["weight"], info["gamma"])
            info["alpha"] = info["gamma"].flatten() * math.sqrt(info["in_f"])
            info["alpha_q15"] = quantize_alpha(info["alpha"])
            info["packed"] = pack_weights(info["wt_ternary"], cols)

    elif args.model:
        layers = load_hf_model(args.model)
        for name, info in layers.items():
            info["wt_ternary"] = ternarize(info["weight"], info["gamma"])
            info["alpha"] = info["gamma"].flatten() * math.sqrt(info["in_f"])
            info["alpha_q15"] = quantize_alpha(info["alpha"])
            info["packed"] = pack_weights(info["wt_ternary"], cols)

    if not layers:
        print("ERROR: No layers extracted.", file=sys.stderr)
        sys.exit(1)

    # The ternarycore HW pipeline (ternary_pipeline.v) has no bias port.
    # Exporting a biased layer would silently drop the bias and produce
    # wrong inference, so refuse loudly instead.
    biased = [n for n, info in layers.items() if info.get("has_bias")]
    if biased:
        print(
            "ERROR: the following layer(s) have a bias term, but the ternarycore\n"
            "  HW pipeline (ternary_pipeline.v) has no bias port and cannot apply it:\n"
            f"    {', '.join(biased)}\n"
            "  Re-export a bias-free model (BitNetLinear typically trains bias-free),\n"
            "  or fold the bias into the preceding layer's weights in software.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Extracted {len(layers)} layers:", file=sys.stderr)
    for name, info in layers.items():
        wt = info.get("wt_ternary", info.get("weight"))
        print(f"  {name}: {wt.shape[0]}x{wt.shape[1]}", file=sys.stderr)

    # For synthetic mode, compute packed and alpha_q15 after the fact
    if args.synthetic:
        for name, info in layers.items():
            info["alpha_q15"] = quantize_alpha(info["alpha"])
            info["packed"] = pack_weights(info["wt_ternary"], cols)

    # ── Generate outputs ──────────────────────────────────────────

    # 1. weights.h
    print(f"\nGenerating {out_dir / 'weights.h'}...", file=sys.stderr)
    header = generate_header(layers, cols)
    (out_dir / "weights.h").write_text(header)

    # 2. layer0_test.cpp
    layer_name = args.layer
    if layer_name is None:
        layer_name = next(iter(layers))
    print(
        f"Generating {out_dir / 'layer0_test.cpp'} for layer '{layer_name}'...",
        file=sys.stderr,
    )
    testbench = generate_testbench(layers, layer_name, cols, depth, seed)
    (out_dir / "layer0_test.cpp").write_text(testbench)

    # 3. metadata.json
    metadata = {
        "format": "ternarycore_weight_enc_v1",
        "weight_encoding": {
            "00": 0,
            "01": "+1",
            "10": "-1",
            "bits_per_weight": 2,
            "weights_per_byte": cols,
            "packing_order": "col0=bits[1:0], col1=bits[3:2], col2=bits[5:4], col3=bits[7:6]",
        },
        "alpha_format": {
            "type": "Q15_unsigned",
            "precision": 15,
            "one_dot_zero": 32768,
            "bits_per_alpha": 16,
        },
        "pipeline_config": {
            "cols": cols,
            "depth": depth,
        },
        "layers": {},
    }
    for name, info in layers.items():
        wt = info.get("wt_ternary", info["weight"])
        metadata["layers"][name] = {
            "shape": list(wt.shape),
            "has_bias": info["has_bias"],
            "nnz": int(np.count_nonzero(wt)),
            "sparsity": float(1.0 - np.count_nonzero(wt) / wt.size),
            "alphas_q15": info.get("alpha_q15", info["alpha"]).flatten().tolist(),
            "packed_bytes": len(info.get("packed", [])),
        }
        # Add alpha_float for reference
        metadata["layers"][name]["alphas_float"] = [
            round(float(v), 6) for v in info["alpha"].flatten()
        ]

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Generating {out_dir / 'metadata.json'}...", file=sys.stderr)

    # Print summary
    print("\n─── Export summary ───", file=sys.stderr)
    total_bytes = sum(len(info.get("packed", [])) for info in layers.values())
    total_fp = sum(info["weight"].size for info in layers.values())
    print(f"  Layers: {len(layers)}", file=sys.stderr)
    print(
        f"  Packed weight bytes: {total_bytes} ({total_fp * 32 / 8 / total_bytes:.0f}x compression vs FP32)",
        file=sys.stderr,
    )
    print(f"  Output directory: {out_dir.resolve()}", file=sys.stderr)
    print(file=sys.stderr)
    print("To build the Verilator testbench:", file=sys.stderr)
    print(
        f"  {build_command(str(out_dir.resolve()), 'layer0_test', cols, depth)}",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print("To run:", file=sys.stderr)
    print(
        f"  cd {shlex.quote(str(out_dir.resolve()))} && ./layer0_test",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
