#!/usr/bin/env python3
"""Lookup-table ternary dot products for packed 2-bit weights.

The table is rebuilt for each activation block.  Each packed byte contains
four values in {-1, 0, +1}; consequently the byte's contribution can be
looked up instead of decoding four weights and branching four times.  This
is the software equivalent of the ternary lookup-table datapath described in
Bitnet.cpp and is also a useful reference for an FPGA implementation.

The public function accepts the hardware layout used by TernaryCore:
``packed[k, group]`` stores four output weights for input element ``k``.
"""
import argparse
import time

import numpy as np


def _tables(activations):
    """Return one 256-entry contribution table per input row."""
    a = np.asarray(activations, dtype=np.int32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = (np.arange(256, dtype=np.uint16)[:, None] >> shifts) & 3
    # 00=0, 01=+1, 10=-1; 11 is reserved and follows RTL semantics (-1).
    signs = np.array([0, 1, -1, -1], dtype=np.int32)
    return (signs[codes][None, :, :] * a[:, None, None]).sum(axis=2)


def dot_packed_lut(activations, packed):
    """Compute ``activations @ weights`` from TernaryCore packed weights.

    ``activations`` has shape ``[depth]`` and ``packed`` has shape
    ``[depth, ceil(outputs / 4)]``.  The result is int64 to avoid silently
    overflowing while validating large model layers.
    """
    a = np.asarray(activations, dtype=np.int32)
    p = np.asarray(packed, dtype=np.uint8)
    if a.ndim != 1 or p.ndim != 2 or p.shape[0] != a.size:
        raise ValueError("expected activations [depth] and packed [depth, groups]")
    table = _tables(a)
    return table[np.arange(a.size)[:, None], p].sum(axis=0, dtype=np.int64)


def unpack(packed):
    """Decode packed bytes to {-1, 0, +1} for a reference comparison."""
    p = np.asarray(packed, dtype=np.uint8)
    codes = (p[..., None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    return np.array([0, 1, -1, -1], dtype=np.int8)[codes]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=1024)
    ap.add_argument("--outputs", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    acts = rng.integers(-127, 128, args.depth, dtype=np.int32)
    weights = rng.integers(-1, 2, (args.depth, args.outputs), dtype=np.int8)
    codes = np.where(weights == 1, 1, np.where(weights == -1, 2, 0)).astype(np.uint8)
    packed = (codes.reshape(args.depth, -1, 4) *
              np.array([1, 4, 16, 64], dtype=np.uint8)).sum(axis=2)
    expected = acts.astype(np.int64) @ weights.astype(np.int64)
    got = dot_packed_lut(acts, packed)
    if not np.array_equal(got, expected):
        raise SystemExit("LUT result does not match decoded reference")
    t0 = time.perf_counter()
    for _ in range(args.reps):
        dot_packed_lut(acts, packed)
    dt = time.perf_counter() - t0
    print(f"{args.depth}x{args.outputs}: {args.reps / dt:.1f} LUT dots/s")


if __name__ == "__main__":
    main()
