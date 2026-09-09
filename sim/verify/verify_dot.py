# verify_dot.py
# Python reference implementation for ternary_dot.
# Computes expected dot products for the same test vectors as tb_ternary_dot.v,
# then runs 10 random 64-element vectors for extended verification.
#
# Run with: python3 sim/verify/verify_dot.py

import random


def to_signed8(val: int) -> int:
    """Convert an unsigned 8-bit integer to signed."""
    return val if val < 128 else val - 256


def ternary_dot(activations: list, weights_enc: list) -> int:
    """Reference dot product over a ternary-weight vector.

    activations: list of signed ints (8-bit range)
    weights_enc: list of 2-bit codes — 0b00=zero, 0b01=+1, 0b10=-1
    """
    acc = 0
    for act, wenc in zip(activations, weights_enc):
        if wenc == 0b00:
            pass
        elif wenc == 0b01:
            acc += act
        else:  # 0b10
            acc -= act
    return acc


# ── Fixed test vectors (matching tb_ternary_dot.v) ──────────────────────────
print("--- Fixed test vectors (VLEN=8, matching Verilog testbench) ---")
errors = 0

fixed_tests = [
    # (description, activations, weights_enc, expected)
    ("all +1, acts 1–8",
     [1, 2, 3, 4, 5, 6, 7, 8],
     [0b01]*8, 36),
    ("all -1, acts 1–8",
     [1, 2, 3, 4, 5, 6, 7, 8],
     [0b10]*8, -36),
    ("all 0 weights",
     [99, 42, 13, 7, 200, 1, 88, 55],
     [0b00]*8, 0),
    ("mixed ±1, act=5 each",
     [5]*8,
     [0b01]*4 + [0b10]*4, 0),
    ("signed act=-5, w=+1 ×8",
     [-5]*8,
     [0b01]*8, -40),
    ("signed act=-5, w=-1 ×8",
     [-5]*8,
     [0b10]*8, 40),
    ("act=127, w=+1 ×8",
     [127]*8,
     [0b01]*8, 1016),
]

for desc, acts, wencs, expected in fixed_tests:
    result = ternary_dot(acts, wencs)
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        errors += 1
    print(f"  {status}: {desc} => {result} (expected {expected})")

# ── Random 64-element vectors ────────────────────────────────────────────────
print("\n--- Random 64-element vectors ---")
random.seed(42)
for trial in range(10):
    acts  = [random.randint(-128, 127) for _ in range(64)]
    wencs = [random.choice([0b00, 0b01, 0b10]) for _ in range(64)]
    result = ternary_dot(acts, wencs)
    print(f"  Trial {trial:2d}: dot product = {result:6d}")

print(f"\n--- {errors} fixed-vector error(s) ---")
if errors == 0:
    print("ALL TESTS PASSED")
else:
    print("FAILURES DETECTED — check test vector definitions")
raise SystemExit(errors)
