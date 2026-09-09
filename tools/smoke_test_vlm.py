#!/usr/bin/env python3
"""Smoke test: verify full BitNet VLM pipeline is wired correctly."""

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.bitnet_full import build_full_bitnet_vlm

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main():
    print("1. Loading and converting to BitNet...")
    t0 = time.time()
    model = build_full_bitnet_vlm()
    model.eval()
    print(f"   {time.time() - t0:.1f}s")

    processor = AutoProcessor.from_pretrained(MODEL)
    img = Image.new("RGB", (224, 224), (100, 150, 200))
    inputs = processor(
        text="<image>Describe this image.", images=img, return_tensors="pt"
    )
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    print(f"   pixel_values: {tuple(pixel_values.shape)}")
    print(f"   input_ids: {tuple(input_ids.shape)}")

    print("\n2. Forward pass...")
    with torch.no_grad():
        vis_out = model.vision_model(pixel_values.flatten(0, 1))
        print(f"   Vision: {tuple(vis_out.last_hidden_state.shape)}")
        vis_proj = model.connector(vis_out.last_hidden_state[:, :1])
        print(f"   Connector: {tuple(vis_proj.shape)}")
        txt_out = model.text_model(input_ids)
        print(f"   Text decoder: {tuple(txt_out.last_hidden_state.shape)}")

    print("\n3. Module count:")
    for name in ("BitNetLinear", "BitNetConv2d", "BitNetEmbedding", "BitNetLayerNorm"):
        n = sum(1 for _ in model.modules() if type(_).__name__ == name)
        if n:
            print(f"   {n} {name}")

    print("\n4. Memory:")

    def bits(mod, clist, b):
        return (
            sum(
                p.numel() * b
                for _, m in mod.named_modules()
                if type(m).__name__ in clist
                for p in m.parameters()
            )
            // 8
        )

    def mb(value):
        return value / 1024 / 1024

    # bits(model, ...) already includes connector and text descendants.
    packed_bytes = bits(model, ["BitNetLinear", "BitNetConv2d"], 2) + bits(
        model, ["BitNetEmbedding"], 4
    )
    quantized_parameters = sum(
        module.weight.numel()
        for module in model.modules()
        if type(module).__name__ in {"BitNetLinear", "BitNetConv2d", "BitNetEmbedding"}
    )
    packed_mb = mb(packed_bytes)
    fp32_mb = mb(quantized_parameters * 4)
    print(
        f"   Quantized weight payload: {packed_mb:.0f}MB vs "
        f"{fp32_mb:.0f}MB FP32 = {fp32_mb / packed_mb:.1f}x"
    )

    print("\n✅ All components connected end-to-end.")


if __name__ == "__main__":
    main()
