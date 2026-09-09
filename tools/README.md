# tools/ — TernaryCore weight tooling

The **Quantum → FPGA bridge**: turn trained model weights into the ternary
`{-1, 0, +1}` codes the hardware runs natively. These scripts are the
"compilation" step — complexity lives here in Python, so the RTL stays elegant
and multiplier-free.

## Weight encoding (shared by all repos)

| 2-bit code | ternary | MAC operation        |
|------------|---------|----------------------|
| `0b00`     | 0       | skip (acc unchanged) |
| `0b01`     | +1      | `acc += activation`  |
| `0b10`     | −1      | `acc -= activation`  |
| `0b11`     | —       | illegal (never emitted) |

Packing is **LSB-first, 2 bits per weight**. A GEMM row of columns
`(+1, −1, +1, 0)` packs to the byte `0x19`, matching `tb/tb_ternary_gemm.v`
`W_row[0]`; `--layout bytes` (4 weights/byte) matches `rtl/weight_bram.v` and
`firmware/tier1_bench.c` `init_weights()`.

## `quantum-mapping.py`

```bash
python3 tools/quantum-mapping.py --demo
python3 tools/quantum-mapping.py --in W.npy --strategy qubo    --report
python3 tools/quantum-mapping.py --in W.npy --strategy quantum --report  # IBM Q if installed
```

Strategies: `absmean` (standard BitNet, default), `qubo` (exact per-weight QUBO),
`quantum` (same QUBO on Qiskit/QAOA, **graceful fallback to exact when qiskit is
absent**), `error-feedback` (experimental sigma-delta). `--report` prints
per-element MSE *and* row-sum residual (baseline vs strategy) — the reproducible
benchmark for articles / the O-1 record. Note: for an independent per-weight
objective the QUBO optimum equals rounding; its value is the Quantum→FPGA bridge,
not a free accuracy win.

## `weights-to-mem.py`

```bash
python3 tools/weights-to-mem.py --demo
python3 tools/weights-to-mem.py --in W.npy --out weights.mem --layout gemm
python3 tools/weights-to-mem.py --in W.npy --layout bytes \
        --emit-c-header firmware/bitnet_weights.h --name bitnet_weights
```

- `--out` → `$readmemh`-ready `.mem` (`gemm` / `bytes` / `flat` layouts).
- `--emit-c-header` → C byte array for firmware BRAM init (see `DEVKIT.md` §5).
- `--activations` → int8 two's-complement hex stream for testbenches.

Inputs may be floats (auto-ternarized via absmean) or already-ternary
`{-1,0,+1}`; formats `.npy/.npz/.csv/.txt`.

## Requirements

Pure-Python core (no numpy needed for `.csv/.txt` or `--demo`). `numpy` only for
`.npy/.npz`. `qiskit` / `qiskit-optimization` optional, only for `--strategy quantum`.

---

## BitNet model tooling

The scripts below convert and evaluate **real** BitNet / VLM models through the
ternarycore pipeline. They use `torch`, `transformers`, and `onnx` (see
[`pyproject.toml`](../pyproject.toml)); install with `uv sync` or `pip install .`.

### `bench.py`

Cross-model quantization benchmark with Pareto search:

```bash
uv run python tools/bench.py --model HuggingFaceTB/SmolVLM-256M-Instruct --qwidth 4
uv run python tools/bench.py --qwidth 4,8 --layers 0,2 --seed 42
```

### `bitnet_full.py`

Full SmolVLM BitNet conversion — ternarizes vision encoder, connector, and
text decoder; INT4/FP4 embedding; per-channel alpha (BitNet b1.58).

### `convert_bitnet_onnx.py`

Synthetic ternarycore pipeline-vector generator. It can validate a real ONNX
file, but ONNX weight extraction is not implemented yet and exits non-zero
instead of claiming a conversion succeeded.

### `convert_smolvlm.py`

SmolVLM-256M analysis helper: download, ternarize linear layers, report weight
statistics, and emit deterministic bit-exact pipeline vectors.

### `eval.py` / `eval_vlm.py`

Eval pipelines for bitnet-quantized SmolVLM: per-layer cosine similarity,
weight-only ternary WikiText-2 perplexity, memory footprint, and VLM captioning
with throughput metrics.

### `export_ternary.py`

Exports trained BitNet weights to hardware format:

```bash
uv run python tools/export_ternary.py   # writes exported/{weights.h, layer0_test.cpp, metadata.json}
```

### `quant_search.py`

Pareto-optimal quantization config search for the ternarycore pipeline
(activation width × clipping method trade-offs).

### `smoke_test_vlm.py`

Fast smoke test that a converted SmolVLM model loads and runs a short caption.

### `train_bitnet.py`

BitNet b1.58 language-model fine-tuning loop (straight-through estimator) with
checkpoint resume support, producing ternary weights consumable by the export
tools above. `--full-model` converts the full VLM but trains the text path only.
