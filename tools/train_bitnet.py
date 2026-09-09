#!/usr/bin/env python3
"""
BitNet b1.58 fine-tuning recipe for SmolVLM-256M.

Implements the BitNet training method:
  1. Initialize ternary weights: W_ternary = clamp(round(W / gamma), -1, 1)
  2. STE forward: Y = (W_ternary @ X) * (gamma * sqrt(dim))
  3. STE backward: dW = dY @ X.T  (pass through the round/clamp)
  4. Update full-precision weights: W -= lr * dW

Train on WikiText-2, evaluate perplexity after each epoch.

Usage:
  uv run python train_bitnet.py                          # train from scratch
  uv run python train_bitnet.py --resume checkpoints/latest.pt
  uv run python train_bitnet.py --full-model             # train full VLM (vision+connector+text)
  uv run python train_bitnet.py --epochs 3 --lr 5e-5
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# ── BitNet linear layer ──────────────────────────────────────────


class BitNetLinear(nn.Module):
    """Linear layer with BitNet b1.58 ternary weights and STE."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision latent weights (what gets trained)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Per-channel alpha scale
        self.gamma = nn.Parameter(torch.ones(out_features))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.gamma.data = self.weight.abs().mean(dim=1).clamp(min=1e-8)

    def ternarize(self):
        """Compute ternary weights with STE."""
        # Forward: quantize
        g = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_ternary = torch.clamp(torch.round(self.weight / g), -1, 1)

        # STE: pass gradient through (no gradient through round/clamp)
        return self.weight + (w_ternary - self.weight).detach()

    def forward(self, x):
        w_t = self.ternarize()
        # Scale: alpha = gamma * sqrt(in_features)
        alpha = self.gamma.unsqueeze(1) * math.sqrt(self.in_features)
        out = F.linear(x, w_t * alpha, self.bias)
        return out


# ── Model conversion (text-decoder-only path) ────────────────────


def convert_to_bitnet(text_model):
    """Replace all Linear layers with BitNetLinear in the text_model."""
    for name, module in list(text_model.named_children()):
        if isinstance(module, nn.Linear):
            bn = BitNetLinear(
                module.in_features, module.out_features, module.bias is not None
            )
            with torch.no_grad():
                bn.weight.copy_(module.weight)
                bn.gamma.data = bn.weight.abs().mean(dim=1).clamp(min=1e-8)
                if module.bias is not None:
                    bn.bias.copy_(module.bias)
            setattr(text_model, name, bn)
        elif len(list(module.children())) > 0:
            convert_to_bitnet(module)
    return text_model


def flatten_model(model, full_model=False):
    """Collect trainable parameters from all BitNet module types.

    When full_model=True, handles all types from bitnet_full:
      BitNetLinear    → weight + gamma
      BitNetConv2d    → weight + gamma
      BitNetEmbedding → weight
      BitNetLayerNorm → weight + bias (when elementwise_affine)

    Otherwise only checks the local BitNetLinear class.
    """
    params = []
    for name, module in model.named_modules():
        tname = type(module).__name__
        if full_model:
            if tname in ("BitNetLinear", "BitNetConv2d"):
                params.append(module.weight)
                params.append(module.gamma)
            elif tname == "BitNetEmbedding":
                params.append(module.weight)
            elif tname == "BitNetLayerNorm" and getattr(
                module, "elementwise_affine", False
            ):
                params.append(module.weight)
                params.append(module.bias)
        else:
            if isinstance(module, BitNetLinear):
                params.append(module.weight)
                params.append(module.gamma)
    return params


# ── Data ──────────────────────────────────────────────────────────

DATASETS = {
    "wikitext-2": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2",
}


def load_dataset(dataset, tokenizer, seq_len=128, split="train", max_samples=5000):
    base_url = DATASETS.get(dataset)
    if base_url is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}"
        )
    url = f"{base_url}/{split}.txt"
    import requests

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    text = r.text.replace(chr(10), " ").replace(chr(13), " ")[: max_samples * 5000]
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len * max(1, max_samples // 5),
    )
    ids = enc.input_ids[0]
    ids = ids[: len(ids) - len(ids) % seq_len]
    if ids.numel() == 0:
        raise ValueError(
            f"dataset split '{split}' did not produce a complete {seq_len}-token sample"
        )
    return ids.view(-1, seq_len)


@torch.no_grad()
def compute_ppl(text_model, embed, input_ids):
    logits = text_model(input_ids).last_hidden_state @ embed.T
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1)
    )
    return math.exp(loss.item())


# ── Training loop ─────────────────────────────────────────────────


def train_epoch(text_submodel, embed, loader, optimizer, scheduler, device, tparams):
    text_submodel.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = text_submodel(batch).last_hidden_state @ embed.T
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(tparams, 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 10 == 0:
            print(f"    batch {n_batches}: loss={loss.item():.4f}")

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="wikitext-2",
        help="Dataset name (default: %(default)s). Available: wikitext-2",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Use bitnet_full.py conversion for all VLM components "
        "(vision encoder + connector + text decoder)",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.seq_len <= 1 or args.batch_size <= 0:
        parser.error(
            "--epochs and --batch-size must be positive; --seq-len must exceed 1"
        )
    if args.samples <= 0 or args.lr <= 0:
        parser.error("--samples and --lr must be positive")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"checkpoint not found: {args.resume}")

    device = (
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        if args.device == "auto"
        else args.device
    )
    print(f"Device: {device}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token

    # ── Load and convert model ──
    if args.full_model:
        from bitnet_full import build_full_bitnet_vlm

        print("Loading and converting full VLM to BitNet...")
        model = build_full_bitnet_vlm()
        model = model.to(device)

        # Freeze vision encoder; train connector + text decoder
        for p in model.vision_model.parameters():
            p.requires_grad = False
        print("Vision encoder frozen.")

        # This recipe is language-model fine-tuning; no image features enter
        # the loop, so training connector parameters here would be a no-op.
        for parameter in model.connector.parameters():
            parameter.requires_grad = False
        tparams = flatten_model(model.text_model, full_model=True)
        n_params = sum(p.numel() for p in tparams)
        print(f"Trainable params (connector + text): {n_params:,}")

        text_submodel = model.text_model
        embed_weight = text_submodel.embed_tokens.weight
    else:
        print(f"Loading {MODEL_ID}...")
        model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        model.eval()

        print("Converting to BitNet...")
        convert_to_bitnet(model.text_model)
        model = model.to(device)

        model.text_model.embed_tokens.weight.requires_grad = False

        tparams = flatten_model(model.text_model)
        n_params = sum(p.numel() for p in tparams)
        print(f"Trainable params (text decoder): {n_params:,}")

        text_submodel = model.text_model
        embed_weight = text_submodel.embed_tokens.weight

    # Freeze everything except the explicitly selected BitNet parameters. This
    # avoids retaining gradients for parameters the optimizer cannot update.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in tparams:
        parameter.requires_grad = True

    # ── Data ──
    print("Loading data...")
    train_data = load_dataset(args.dataset, tok, args.seq_len, "train", args.samples)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = load_dataset(
        args.dataset, tok, args.seq_len, "test", min(args.samples // 5, 200)
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(tparams, lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    start_epoch = 1
    best_ppl = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        target = model if args.full_model else model.text_model
        target.load_state_dict(state, strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_ppl = float(checkpoint.get("ppl", best_ppl))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    # ── Training ──
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            text_submodel, embed_weight, loader, optimizer, scheduler, device, tparams
        )

        # Evaluate
        text_submodel.eval()
        with torch.no_grad():
            ppl = compute_ppl(text_submodel, embed_weight, val_data.to(device))

        t1 = time.time()
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_ppl={ppl:.1f} "
            f"({t1 - t0:.0f}s)"
        )

        # Save checkpoint
        if args.full_model:
            state_dict = model.state_dict()
        else:
            state_dict = model.text_model.state_dict()

        ckpt = args.output / f"bitnet_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ppl": ppl,
                "full_model": args.full_model,
            },
            ckpt,
        )
        print(f"  Saved {ckpt}")

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(state_dict, args.output / "best.pt")

    print(f"\nDone! Best validation PPL: {best_ppl:.1f}")


if __name__ == "__main__":
    main()
