#!/usr/bin/env python3
# verify_gemm_simple.py
# Simple Python reference implementation for ternary_gemm without numpy.
# Verifies the 4x4 test case from tb_ternary_gemm.v


def ternary_weight_decode(enc):
    """Decode ternary weight encoding: 0=zero, 1=+1, 2=-1"""
    if enc == 0:
        return 0
    elif enc == 1:
        return 1
    elif enc == 2:
        return -1
    else:
        raise ValueError(f"Invalid weight encoding: {enc}")


def ternary_gemm_ref(A, W_enc):
    """Reference 2D ternary GEMM: C = A * W.

    A      — (ROWS, DEPTH) int8 activation matrix
    W_enc  — (DEPTH, COLS) ternary weight encoding matrix
             values: 0=zero, 1=+1, 2=-1
    Returns C — (ROWS, COLS) int32 output matrix
    """
    ROWS = len(A)
    DEPTH = len(A[0])
    COLS = len(W_enc[0])

    # Decode weights
    W = [
        [ternary_weight_decode(W_enc[k][c]) for c in range(COLS)] for k in range(DEPTH)
    ]

    # Compute matrix multiplication
    C = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    for r in range(ROWS):
        for c in range(COLS):
            acc = 0
            for k in range(DEPTH):
                acc += A[r][k] * W[k][c]
            C[r][c] = acc

    return C


# ── Fixed 4×4 test (matching tb_ternary_gemm.v) ─────────────────────────────
print("--- Fixed 4×4 test (matching Verilog testbench) ---")

A = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [-1, -2, -3, -4],
    [10, 0, 5, -3],
]

# W weights: 0=zero, 1=+1, 2=-1
# W = [[+1,-1,+1,0],[0,+1,-1,+1],[-1,0,+1,-1],[+1,-1,0,+1]]
W_enc = [
    [1, 2, 1, 0],
    [0, 1, 2, 1],
    [2, 0, 1, 2],
    [1, 2, 0, 1],
]

C = ternary_gemm_ref(A, W_enc)

C_expected = [
    [2, -3, 2, 3],
    [6, -7, 6, 7],
    [-2, 3, -2, -3],
    [2, -7, 15, -8],
]

errors = 0
for r in range(4):
    for c in range(4):
        if C[r][c] != C_expected[r][c]:
            print(f"FAIL row={r} col={c}: got={C[r][c]} expected={C_expected[r][c]}")
            errors += 1

if errors == 0:
    print(f"PASS: All 4x4 test vectors match expected results")
    print(f"Result matrix C:")
    for r in range(4):
        print(f"  Row {r}: {C[r]}")
else:
    print(f"--- {errors} error(s) ---")

print("\n--- Verification complete ---")
raise SystemExit(errors)
