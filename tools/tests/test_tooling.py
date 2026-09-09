import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import convert_bitnet_onnx as onnx_converter
import export_ternary as exporter
import hardware_semantics as semantics


class ExporterTests(unittest.TestCase):
    def test_pack_weights_matches_hardware_layout(self):
        weights = np.array([[1], [-1], [1], [0]], dtype=np.int8)
        self.assertEqual(exporter.pack_weights(weights).tolist(), [0x19])
        self.assertEqual(semantics.pack_ternary_weights([1, -1, 1, 0], 4), 0x19)

    def test_pack_weights_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, r"only -1, 0, or \+1"):
            exporter.pack_weights(np.array([[2]], dtype=np.int8), cols=1)
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            exporter.pack_weights(np.zeros((5, 1), dtype=np.int8), cols=5)

    def test_synthetic_generation_is_seeded_and_supports_full_sparsity(self):
        first, _ = exporter.gen_synthetic_weights(4, 8, sparsity=1.0, seed=7)
        second, _ = exporter.gen_synthetic_weights(4, 8, sparsity=1.0, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.any(first))

    def test_non_finite_alpha_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            exporter.quantize_alpha(np.array([np.nan]))

    def test_build_command_includes_shared_weight_cell_and_quotes_paths(self):
        command = exporter.build_command("/tmp/output with spaces")
        self.assertIn("ternary_weight.v", command)
        self.assertIn("'/tmp/output with spaces/layer0_test.cpp'", command)

    def test_generated_result_capture_supports_all_packed_column_widths(self):
        for cols in range(1, 5):
            weights, alphas = exporter.gen_synthetic_weights(
                cols, 4, sparsity=0.5, seed=cols
            )
            layers = {
                "synthetic": {
                    "wt_ternary": weights,
                    "alpha_q15": exporter.quantize_alpha(alphas),
                }
            }
            source = exporter.generate_testbench(
                layers, "synthetic", cols=cols, depth=4
            )
            self.assertIn("if (dut->valid_out)", source)
            if cols <= 2:
                self.assertNotIn("dut->result.at(c)", source)
            else:
                self.assertIn("dut->result.at(c)", source)


class OnnxVectorTests(unittest.TestCase):
    def test_packing_rejects_missing_or_illegal_weights(self):
        with self.assertRaisesRegex(ValueError, "expected at least"):
            onnx_converter.pack_ternary_weights([1], cols=2)
        with self.assertRaisesRegex(ValueError, r"only -1, 0, or \+1"):
            onnx_converter.pack_ternary_weights([3], cols=1)

    def test_generated_verilog_uses_sized_literals(self):
        tests = onnx_converter.generate_tests(1, 4, rng_seed=9)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tb.v"
            onnx_converter.write_verilog_tb(tests, output)
            source = output.read_text()
        self.assertIn("activation = -8'sd", source)
        self.assertIn("inv = 22'd", source)
        self.assertIn("weight_enc = 8'h", source)
        self.assertIn("alpha = 64'h", source)


if __name__ == "__main__":
    unittest.main()
