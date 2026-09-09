#!/usr/bin/env python3
"""
VLM Image Evaluation Pipeline for BitNet-quantized SmolVLM-256M.

Loads a BitNet-converted (or raw FP32) SmolVLMForConditionalGeneration model,
generates captions for synthetic or saved images, and reports throughput and
memory metrics.

Usage:
  uv run python tools/eval_vlm.py --images 5 --synthetic
  uv run python tools/eval_vlm.py --images 3 --checkpoint checkpoints/best.pt
  uv run python tools/eval_vlm.py --images 3 --load-bitnet
  uv run python tools/eval_vlm.py --images 3 --fp32                           # baseline comparison
  uv run python tools/eval_vlm.py --images 3 --checkpoint checkpoints/best.pt --max-new-tokens 30
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitnet_full import (
    BitNetConv2d,
    BitNetEmbedding,
    BitNetLayerNorm,
    BitNetLinear,
    convert_module,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "vlm_eval" / "images"

# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------


def _make_gradient(w, h, c1, c2):
    img = Image.new("RGB", (w, h))
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw = ImageDraw.Draw(img)
        draw.line([(0, y), (w - 1, y)], fill=(r, g, b))
    return img


def generate_synthetic_images(count=20, size=(512, 512), seed=42):
    """Return (images, labels) list of synthetic PIL Images."""
    torch.manual_seed(seed)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 128, 0),
        (0, 128, 128),
        (128, 128, 0),
        (128, 0, 0),
        (0, 128, 0),
    ]
    shapes = [
        "circle",
        "square",
        "triangle",
        "star",
        "diamond",
        "rectangle",
        "oval",
        "hexagon",
        "cross",
        "arrow",
        "heart",
        "ring",
        "trapezoid",
        "pentagon",
        "crescent",
        "chevron",
        "spiral",
        "wave",
        "semicircle",
        "parallelogram",
    ]
    images, labels = [], []
    for i in range(count):
        img = _make_gradient(size[0], size[1], (240, 240, 240), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        cx, cy = size[0] // 2, size[1] // 2
        r = size[0] // 3
        fg = colors[(i + 6) % len(colors)]
        shape = shapes[i % len(shapes)]

        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg, outline=(0, 0, 0))
        elif shape == "oval":
            draw.ellipse(
                [cx - r, cy - r // 2, cx + r, cy + r // 2], fill=fg, outline=(0, 0, 0)
            )
        elif shape == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg, outline=(0, 0, 0))
        elif shape == "rectangle":
            draw.rectangle(
                [cx - r, cy - r // 2, cx + r, cy + r // 2], fill=fg, outline=(0, 0, 0)
            )
        elif shape == "triangle":
            draw.polygon(
                [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
                fill=fg,
                outline=(0, 0, 0),
            )
        elif shape == "star":
            pts = []
            for j in range(5):
                a1 = math.radians(-90 + j * 72)
                a2 = math.radians(-90 + j * 72 + 36)
                pts.append((cx + r * math.cos(a1), cy + r * math.sin(a1)))
                pts.append((cx + r * 0.4 * math.cos(a2), cy + r * 0.4 * math.sin(a2)))
            draw.polygon(pts, fill=fg, outline=(0, 0, 0))
        elif shape == "diamond":
            draw.polygon(
                [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                fill=fg,
                outline=(0, 0, 0),
            )
        elif shape == "ring":
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r], outline=fg, width=size[0] // 20
            )
        elif shape == "cross":
            w = size[0] // 10
            draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=fg)
            draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=fg)
        elif shape == "arrow":
            pts = [
                (cx - r, cy),
                (cx, cy - r),
                (cx, cy - r // 3),
                (cx + r, cy - r // 3),
                (cx + r, cy + r // 3),
                (cx, cy + r // 3),
                (cx, cy + r),
            ]
            draw.polygon(pts, fill=fg, outline=(0, 0, 0))
        else:
            # fallback: filled circle with label
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg, outline=(0, 0, 0))

        # overlay gradient blobs
        for _ in range(3):
            bx = torch.randint(0, size[0], (1,)).item()
            by = torch.randint(0, size[1], (1,)).item()
            br = torch.randint(size[0] // 8, size[0] // 4, (1,)).item()
            bc = colors[torch.randint(0, len(colors), (1,)).item()]
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=bc, outline=None)

        label = f"img_{i:03d}_{shape}"
        images.append(img)
        labels.append(label)
    return images, labels


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _safe_load_bitnet_state(target_model, ckpt_path, device="cpu"):
    """Load a checkpoint saved from build_full_bitnet_vlm (base SmolVLMModel)
    into a SmolVLMForConditionalGeneration.  Prepends 'model.' to the three
    top-level groups and skips lm_head keys."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    # unwrap training wrapper
    if "model_state" in raw:
        raw = raw["model_state"]

    target_keys = set(target_model.state_dict())
    remapped = {}
    for k, v in raw.items():
        candidates = [k]
        if k.startswith(("vision_model.", "connector.", "text_model.")):
            candidates.append("model." + k)
        else:
            # Text-only checkpoints are saved from model.text_model.state_dict().
            candidates.append("model.text_model." + k)
        match = next(
            (candidate for candidate in candidates if candidate in target_keys), None
        )
        if match is None:
            print(
                f"  Skipping checkpoint key (no match in ForConditionalGeneration): {k}"
            )
        else:
            remapped[match] = v

    missing, unexpected = target_model.load_state_dict(remapped, strict=False)
    if missing:
        # Only report non-lm_head missing keys if they matter
        missing_no_lm = [k for k in missing if not k.startswith("lm_head.")]
        if missing_no_lm:
            print(f"  Missing keys (non-lm_head): {len(missing_no_lm)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    ckpt_mb = sum(v.numel() * v.element_size() for v in remapped.values()) / 1024 / 1024
    print(f"  Loaded {len(remapped)} tensors ({ckpt_mb:.0f} MB) from {ckpt_path.name}")
    return target_model


def load_model(
    mode="fp32",
    checkpoint=None,
    qfmt="int4",
    device=None,
):
    """Load SmolVLMForConditionalGeneration in one of several modes.

    Args:
        mode: 'fp32' (raw pretrained), 'bitnet' (fresh conversion), 'checkpoint'.
        checkpoint: Path to a .pt checkpoint (only when mode='checkpoint').
        qfmt: 'int4' or 'fp4' for embedding quantization.
        device: Torch device string.
    Returns:
        (model, processor, model_type_str)
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading {MODEL_ID} (mode={mode})...")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()

    if mode == "fp32":
        model = model.to(device)
        model_type = "FP32 baseline"
    elif mode == "bitnet":
        print("Converting all modules to BitNet b1.58...")
        convert_module(model, qfmt)
        model = model.to(device)
        model_type = f"BitNet (fresh, qfmt={qfmt})"
    elif mode == "checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint path required when mode='checkpoint'")
        print("Converting all modules to BitNet b1.58...")
        # We must convert first so the state dict structure matches
        # (BitNetLinear has .weight+.gamma instead of just .weight)
        convert_module(model, qfmt)
        model = _safe_load_bitnet_state(model, Path(checkpoint), device)
        model = model.to(device)
        model_type = f"BitNet (checkpoint: {Path(checkpoint).name})"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Memory report
    with torch.no_grad():
        mem_fp32 = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
        if mode == "fp32":
            mem_effective = mem_fp32
        else:
            converted_ids = set()
            packed_bits = 0
            for module in model.modules():
                if isinstance(module, (BitNetLinear, BitNetConv2d)):
                    packed_bits += module.weight.numel() * 2
                    converted_ids.add(id(module.weight))
                    packed_bits += module.gamma.numel() * 32
                    converted_ids.add(id(module.gamma))
                    if module.bias is not None:
                        packed_bits += module.bias.numel() * 32
                        converted_ids.add(id(module.bias))
                elif isinstance(module, BitNetEmbedding):
                    packed_bits += module.weight.numel() * 4
                    converted_ids.add(id(module.weight))
                elif isinstance(module, BitNetLayerNorm) and module.elementwise_affine:
                    packed_bits += module.weight.numel() * 8
                    packed_bits += module.bias.numel() * 8
                    converted_ids.update((id(module.weight), id(module.bias)))

            rest_bits = sum(
                parameter.numel() * 32
                for parameter in model.parameters()
                if id(parameter) not in converted_ids
            )
            mem_packed = packed_bits / 8 / 1024 / 1024
            mem_rest = rest_bits / 8 / 1024 / 1024
            mem_effective = mem_packed + mem_rest
            print(
                f"  Memory breakdown: packed {mem_packed:.0f} MB + "
                f"remaining FP32 {mem_rest:.0f} MB = {mem_effective:.0f} MB"
            )

    return model, processor, model_type, mem_fp32, mem_effective


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "<|im_start|>user\n<image>\nDescribe this image in detail.<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_caption(
    model, processor, image, prompt=PROMPT_TEMPLATE, max_new_tokens=50, device=None
):
    """Run greedy caption generation on a single image.

    Returns (caption_text, encode_time_s, decode_time_s, num_tokens).
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Process input
    t0 = time.perf_counter()
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    t_proc = time.perf_counter() - t0

    # Generate
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    t_decode = time.perf_counter() - t0

    num_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    caption = processor.decode(out[0], skip_special_tokens=True)
    # Strip the prompt from the decoded caption
    # The decode includes the full conversation; extract just the assistant reply
    reply = caption.split("<|im_start|>assistant")[-1].strip()
    reply = reply.removeprefix("\n")
    if reply.endswith("<|im_end|>"):
        reply = reply[: -len("<|im_end|>")].strip()

    return reply, t_proc, t_decode, num_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _bar(label, val, unit="", width=20):
    """Simple text bar chart."""
    n = min(int(val), width)
    bar = "█" * n + "░" * (width - n)
    return f"  {label:>16} {bar} {val:.1f}{unit}"


def main():
    parser = argparse.ArgumentParser(description="BitNet VLM image captioning eval")
    parser.add_argument(
        "--images", type=int, default=5, help="Number of images to evaluate"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50, help="Max tokens per caption"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic images instead of real ones",
    )
    parser.add_argument(
        "--image-dir", type=Path, default=IMAGE_DIR, help="Directory with real images"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fp32", action="store_true", help="FP32 baseline (no quantization)"
    )
    group.add_argument(
        "--load-bitnet", action="store_true", help="Fresh BitNet conversion"
    )
    group.add_argument(
        "--checkpoint", type=Path, default=None, help="Path to .pt checkpoint"
    )
    parser.add_argument("--qfmt", default="int4", choices=["int4", "fp4"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.images <= 0:
        parser.error("--images must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    # Determine mode
    if args.fp32:
        mode = "fp32"
    elif args.checkpoint:
        mode = "checkpoint"
    elif args.load_bitnet:
        mode = "bitnet"
    else:
        mode = "fp32"  # default baseline

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print()

    # ── 1. Load model ──────────────────────────────────────────────
    model, processor, model_type, mem_fp32, mem_effective = load_model(
        mode=mode,
        checkpoint=args.checkpoint,
        qfmt=args.qfmt,
        device=device,
    )
    print(f"  Model: {model_type}")
    print(f"  FP32 footprint:    {mem_fp32:.0f} MB")
    print(f"  Effective memory:  {mem_effective:.0f} MB")
    if mem_fp32 > 0 and mem_effective > 0:
        print(f"  Compression ratio: {mem_fp32 / mem_effective:.1f}x")
    print()

    # ── 2. Get images ──────────────────────────────────────────────
    if args.synthetic:
        print(f"Generating {args.images} synthetic images...")
        images, labels = generate_synthetic_images(args.images, seed=args.seed)
    else:
        os.makedirs(args.image_dir, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts)[
            : args.images
        ]
        if len(paths) < args.images:
            print(
                f"Found {len(paths)} images in {args.image_dir}, generating "
                f"{args.images - len(paths)} synthetic images to pad..."
            )
            real_imgs = [Image.open(p).convert("RGB") for p in paths]
            real_labels = [p.stem for p in paths]
            syn_imgs, syn_labels = generate_synthetic_images(
                args.images - len(paths), seed=args.seed
            )
            images = real_imgs + syn_imgs
            labels = real_labels + syn_labels
        else:
            images = [Image.open(p).convert("RGB") for p in paths]
            labels = [p.stem for p in paths]
    print(f"  {len(images)} images loaded")
    print()

    # ── 3. Warmup ──────────────────────────────────────────────────
    # First call is always slow (MPS compile / GPU warmup); do throwaway
    # generation on image 0 so timing starts clean on the real loop.
    print("Warmup...")
    _ = generate_caption(
        model,
        processor,
        images[0],
        prompt=PROMPT_TEMPLATE,
        max_new_tokens=5,
        device=device,
    )
    print()

    # ── 4. Evaluate ────────────────────────────────────────────────
    print("─── Running captioning ───")
    print(
        f"{'#':>3} {'Label':>20} {'Encode':>8} {'Decode':>8} {'Tokens':>7} {'Tokens/s':>10}"
    )
    print("-" * 60)

    all_encode, all_decode, all_tokens = [], [], []
    sample_captions = []

    for idx, (img, label) in enumerate(zip(images, labels)):
        reply, t_proc, t_decode, n_tok = generate_caption(
            model,
            processor,
            img,
            prompt=PROMPT_TEMPLATE,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        all_encode.append(t_proc)
        all_decode.append(t_decode)
        all_tokens.append(n_tok)

        tok_s = n_tok / t_decode if t_decode > 0 else 0.0
        print(
            f"{idx:>3} {label:>20} {t_proc * 1000:>7.0f}ms {t_decode * 1000:>7.0f}ms "
            f"{n_tok:>5}  {tok_s:>8.1f}"
        )

        if idx < 5:
            sample_captions.append((label, reply))

    # ── 4. Report ──────────────────────────────────────────────────
    print()
    print("─── Sample captions ───")
    for label, caption in sample_captions:
        cap_short = caption[:120] + "..." if len(caption) > 120 else caption
        print(f"  [{label}] {cap_short}")

    print()
    print("─── Summary ───")
    avg_encode = sum(all_encode) / len(all_encode)
    avg_decode = sum(all_decode) / len(all_decode)
    total_tokens = sum(all_tokens)
    total_time = sum(all_decode)
    avg_tok_s = total_tokens / total_time if total_time > 0 else 0.0
    avg_img_s = len(images) / total_time if total_time > 0 else 0.0

    print(f"  Images evaluated:  {len(images)}")
    print(f"  Model type:        {model_type}")
    print(_bar("Avg encode", avg_encode * 1000, " ms"))
    print(_bar("Avg decode", avg_decode * 1000, " ms"))
    print(_bar("Avg tokens/sec", avg_tok_s))
    print(_bar("Avg images/sec", avg_img_s))
    print(f"  Total decode time: {total_time:.2f}s")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  FP32 memory:       {mem_fp32:.0f} MB")
    print(f"  Effective memory:  {mem_effective:.0f} MB")
    if mode != "fp32":
        print(f"  Compression:       {mem_fp32 / max(mem_effective, 1):.1f}x")


if __name__ == "__main__":
    main()
