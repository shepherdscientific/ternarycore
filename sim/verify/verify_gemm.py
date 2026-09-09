# verify_gemm.py
# Python / NumPy reference implementation for ternary_gemm.
# Verifies the 4x4 test case from tb_ternary_gemm.v and runs random GEMMs.
#
# Run with: python3 sim/verify/verify_gemm.py

import random
import numpy as np


def ternary_gemm_ref(A: np.ndarray, W_enc: np.ndarray) -> np.ndarray:
    """Reference 2D ternary GEMM: C = A * W.

    A      — (ROWS, DEPTH) int8 activation matrix
    W_enc  — (DEPTH, COLS) ternary weight encoding matrix
             values: 0=zero, 1=+1, 2/3=-1 (RTL-compatible)
    Returns C — (ROWS, COLS) int32 output matrix
    """
    W = np.where(W_enc == 0, 0, np.where(W_enc == 1, 1, -1)).astype(np.int32)
    return A.astype(np.int32) @ W


# ── Fixed 4×4 test (matching tb_ternary_gemm.v) ─────────────────────────────
print("--- Fixed 4×4 test (matching Verilog testbench) ---")

A = np.array([
    [  1,  2,  3,  4],
    [  5,  6,  7,  8],
    [ -1, -2, -3, -4],
    [ 10,  0,  5, -3],
], dtype=np.int8)

# W weights: 0=zero, 1=+1, 2=-1
# W = [[+1,-1,+1,0],[0,+1,-1,+1],[-1,0,+1,-1],[+1,-1,0,+1]]
W_enc = np.array([
    [1, 2, 1, 0],
    [0, 1, 2, 1],
    [2, 0, 1, 2],
    [1, 2, 0, 1],
], dtype=np.int32)

C = ternary_gemm_ref(A, W_enc)

C_expected = np.array([
    [ 2, -3,  2,  3],
    [ 6, -7,  6,  7],
    [-2,  3, -2, -3],
    [ 2, -7, 15, -8],
], dtype=np.int32)

errors = 0
for r in range(4):
    for c in range(4):
        status = "PASS" if C[r, c] == C_expected[r, c] else "FAIL"
        if C[r, c] != C_expected[r, c]:
            errors += 1
        print(f"  {status}: C[{r}][{c}] = {C[r, c]:4d}  (expected {C_expected[r, c]:4d})")

print(f"\n  Computed matrix C:\n{C}\n")
print(f"  Expected matrix C:\n{C_expected}\n")
print(f"  Match: {np.array_equal(C, C_expected)}")

# ── Reserved weight encoding regression ─────────────────────────────────────
print("\n--- Reserved weight encoding 11 ---")
A_reserved = np.array([[127]], dtype=np.int8)
W_reserved = np.array([[3]], dtype=np.int32)
C_reserved = ternary_gemm_ref(A_reserved, W_reserved)
reserved_expected = np.array([[-127]], dtype=np.int32)
reserved_status = "PASS" if np.array_equal(C_reserved, reserved_expected) else "FAIL"
print(
    f"  {reserved_status}: encoding 11 -> {C_reserved[0, 0]} "
    f"(expected {reserved_expected[0, 0]})"
)
if not np.array_equal(C_reserved, reserved_expected):
    errors += 1

# ── Random GEMMs ─────────────────────────────────────────────────────────────
print("\n--- Random 4×4 GEMMs (10 trials, seed=42) ---")
random.seed(42)
np.random.seed(42)

for trial in range(10):
    A_r = np.random.randint(-128, 128, size=(4, 4), dtype=np.int8)
    W_r = np.random.choice([0, 1, 2], size=(4, 4))
    C_r = ternary_gemm_ref(A_r, W_r)
    print(f"  Trial {trial:2d}: C row sums = {C_r.sum(axis=1).tolist()}")

print(f"\n--- {errors} fixed-test error(s) ---")
if errors == 0:
    print("ALL TESTS PASSED")
