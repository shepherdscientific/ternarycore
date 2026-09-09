"""Bit-exact scalar semantics shared by tooling and generated test vectors."""

TERNARY_ENCODING = {0: 0, 1: 1, -1: 2}


def pack_ternary_weights(weights: list[int], cols: int) -> int:
    """Pack up to four ternary values into the RTL weight encoding."""
    if not 1 <= cols <= 4:
        raise ValueError("cols must be between 1 and 4")
    if len(weights) < cols:
        raise ValueError(f"expected at least {cols} weights, got {len(weights)}")
    selected = weights[:cols]
    if any(weight not in TERNARY_ENCODING for weight in selected):
        raise ValueError("weights must contain only -1, 0, or +1")
    return sum(TERNARY_ENCODING[weight] << (2 * index) for index, weight in enumerate(selected))


def quantize_activation(value: int, inv: int, precision: int = 15, q_max: int = 127) -> int:
    """Mirror ``activation_quant.v`` for one signed activation."""
    shifted = (value * inv + (1 << (precision - 1))) >> precision
    return max(-q_max, min(q_max, shifted))


def scale_accumulator(acc: int, alpha: int, precision: int = 15) -> int:
    """Mirror ``ternary_scale.v`` arithmetic shift and upward rounding."""
    product = acc * alpha
    remainder = product & ((1 << precision) - 1)
    return (product >> precision) + int(remainder != 0)
