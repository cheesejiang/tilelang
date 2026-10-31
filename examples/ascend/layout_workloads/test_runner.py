"""Host-only checks; no TileLang/CANN import is needed for report tests."""

import ast
from collections import Counter
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from catalog import CASES, iteration_count, mode_configs, validate
import run_workloads


class ReportTests(unittest.TestCase):
    def test_mode_switch_matrix(self):
        expected = {"baseline": (False, False), "ownership": (True, False),
                    "padding": (False, True), "auto": (True, True)}
        for mode, flags in expected.items():
            config = mode_configs(mode)
            self.assertEqual((config["tl.ascend_enable_ub_ownership"], config["tl.ascend_enable_ub_padding"]), flags)

    def test_source_provenance(self):
        self.assertEqual(sorted(Counter(c.category for c in CASES.values()).values()), [2, 2, 2, 2, 2])
        root = Path(__file__).resolve().parents[3]
        for case in CASES.values():
            tree = ast.parse((root / case.source).read_text())
            functions = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for symbol in case.symbol.split(" + "):
                self.assertIn(symbol.split("(")[0], functions)

    def test_flash_iterations_and_partial_cost_scope(self):
        self.assertEqual(iteration_count("flash_decode_column_store", 64, 4), 64)
        self.assertEqual(iteration_count("flash_decode_qk_column_store", 64, 4), 64)
        self.assertEqual(iteration_count("attention_update", 64, 4), 4)
        from layout_diagnostics import summary_fields
        records = [{"stage": "final", "cost_known": 0, "ub_extra_storage_bytes": 96},
                   {"stage": "padding_scored_final", "cost_known": 1, "entries": 256,
                    "cost_scope": "padding_scored_ops_only_invariant_ops_excluded"}]
        result = summary_fields(records)
        self.assertEqual(result["entries"], 256)
        self.assertEqual(result["ub_extra_storage_bytes"], 96)
        self.assertIn("excluded", result["cost_scope"])
        self.assertIn("reducer", result["unmodeled_costs"])

    def test_invalid_dimensions(self):
        validate(32, 32, 32, 32, 4)
        for columns in (16, 17, 34, 40):
            validate(32, columns, 32, 32, 4)
        for dimensions in [(32, 0, 32, 32, 4), (16, 32, 32, 32, 4), (32, 32, 16, 32, 4), (32, 32, 32, 0, 4)]:
            with self.assertRaises(ValueError):
                validate(*dimensions)
        with self.assertRaisesRegex(ValueError, "256 rows"):
            validate(512, 32, 32, 1, 4, case="flash_decode_qk_column_store")

    def test_suite_collects_all_modes_and_final_allocations(self):
        with tempfile.TemporaryDirectory() as folder:
            def worker(cmd, check):
                mode = cmd[cmd.index("--mode") + 1]
                self.assertTrue((Path(folder) / "summary.md").exists())
                if mode == "ownership":
                    self.assertIn("baseline", (Path(folder) / "summary.md").read_text())
                directory = Path(folder) / "vecadd"
                directory.mkdir(exist_ok=True)
                size = 4192 if mode == "auto" else 4096
                report = {"status": "measured", "latency_us": 8.0 if mode in ("auto", "ownership") else 10.0,
                          "correctness_passed": True,
                          "final_ir_ub": {"status": "known", "ub_allocation_bytes": size},
                          "layout_diagnostics": []}
                (directory / f"{mode}.json").write_text(json.dumps(report))
                return type("Result", (), {"returncode": 0})()

            argv = ["run_workloads.py", "--case", "vecadd", "--bench", "--output", folder]
            with patch("sys.argv", argv), patch("run_workloads.subprocess.run", side_effect=worker), redirect_stdout(io.StringIO()):
                run_workloads.main()
            with (Path(folder) / "summary.csv").open() as handle:
                rows = {row["mode"]: row for row in csv.DictReader(handle)}
            self.assertEqual(set(rows), {"baseline", "ownership", "padding", "auto"})
            self.assertEqual(float(rows["auto"]["speedup_vs_baseline"]), 1.25)
            self.assertEqual(rows["auto"]["final_ir_ub_extra_vs_baseline_bytes"], "96")
            self.assertEqual(rows["baseline"]["iterations"], "1")
            self.assertEqual(rows["auto"]["correctness_passed"], "True")
            self.assertIn("| auto | passed |", (Path(folder) / "summary.md").read_text())

    def test_failed_mode_does_not_prevent_other_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            visited = []

            def worker(cmd, check):
                mode = cmd[cmd.index("--mode") + 1]
                visited.append(mode)
                directory = Path(folder) / "gemm_scale"
                directory.mkdir(exist_ok=True)
                report = {"status": "failed" if mode == "padding" else "correctness_passed"}
                (directory / f"{mode}.json").write_text(json.dumps(report))
                return type("Result", (), {"returncode": int(mode == "padding")})()

            argv = ["run_workloads.py", "--case", "gemm_scale", "--output", folder]
            with patch("sys.argv", argv), patch("run_workloads.subprocess.run", side_effect=worker), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_workloads.main()
            self.assertEqual(visited, ["baseline", "ownership", "padding", "auto"])
            self.assertTrue((Path(folder) / "summary.md").exists())

    def test_flash_summary_keeps_partial_cost_scope_and_lane_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            def worker(cmd, check):
                mode = cmd[cmd.index("--mode") + 1]
                directory = Path(folder) / "flash_decode_column_store"
                directory.mkdir(exist_ok=True)
                report = {"status": "correctness_passed", "correctness_passed": True,
                          "iterations": 32, "column_store_audit": {
                              "status": "known", "min_active_lanes": 1, "max_active_lanes": 1,
                              "max_distinct_lines_per_group": 1, "conflicting_batches": 0},
                          "layout_diagnostics": [
                              {"stage": "final", "cost_known": 0},
                              {"stage": "padding_scored_final", "cost_known": 1, "entries": 32,
                               "cost_scope": "padding_scored_ops_only_invariant_ops_excluded"}]}
                (directory / f"{mode}.json").write_text(json.dumps(report))
                return type("Result", (), {"returncode": 0})()

            argv = ["run_workloads.py", "--case", "flash_decode_column_store", "--output", folder]
            with patch("sys.argv", argv), patch("run_workloads.subprocess.run", side_effect=worker), redirect_stdout(io.StringIO()):
                run_workloads.main()
            with (Path(folder) / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["iterations"] == "32" and row["entries"] == "32" for row in rows))
            self.assertTrue(all("excluded" in row["cost_scope"] for row in rows))
            self.assertIn("Lowered column-store audit", (Path(folder) / "summary.md").read_text())

    def test_summarize_existing_json_without_running_kernels(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "attention_update"
            directory.mkdir()
            archived = json.dumps({"status": "measured", "columns": 128, "rows": 64, "threads": 64,
                                   "blocks": 8, "iterations": 7, "latency_us": 3.5})
            (directory / "auto.json").write_text(archived)
            argv = ["run_workloads.py", "--summarize-only", "--output", folder]
            with patch("sys.argv", argv), patch("run_workloads.subprocess.run") as worker, redirect_stdout(io.StringIO()):
                run_workloads.main()
            worker.assert_not_called()
            with (Path(folder) / "summary.csv").open() as handle:
                (row,) = list(csv.DictReader(handle))
            self.assertEqual((row["columns"], row["blocks"], row["iterations"]), ("128", "8", "7"))
            self.assertEqual((directory / "auto.json").read_text(), archived)


class ReferenceTests(unittest.TestCase):
    def test_cpu_equations(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        from reference import reference
        a = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        b = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / 10
        stats = np.array([[[0.5, 0.75, 0.9], [0.6, 0.8, 0.7], [1, 2, 3]]], dtype=np.float32)
        for case in CASES:
            actual = reference(case, a, b, stats)
            if case == "flash_decode_qk_column_store":
                expected = np.array([[[max(sum(float(a[0, r, d]) * float(b[0, k, n, d])
                                               for d in range(4)) for n in range(3))
                                       for k in range(2)] for r in range(3)]], dtype=np.float32)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
                continue
            if case == "flash_decode_column_store":
                self.assertEqual(actual.shape, (1, 3, 2))
                for r in range(3):
                    for k in range(2):
                        self.assertEqual(float(actual[0, r, k]), max(float(b[0, k, r, n]) for n in range(4)))
                continue
            for r in range(3):
                for c in range(4):
                    x = float(a[0, r, c])
                    if case == "kda_column_store":
                        expected = stats[0, 0, r] if c == 0 else x
                    elif case == "kda_reduce_column_store":
                        expected = sum(float(a[0, r, d]) * float(b[0, 0, r, d]) for d in range(4)) if c == 0 else x
                    elif case == "gemm_scale":
                        expected = 1.25 * x
                    elif case == "attention_finalize":
                        expected = x / stats[0, -1, r]
                    elif case == "vecadd":
                        expected = x + b[0, 0, r, c]
                    elif case == "gemm_bias":
                        expected = x + b[0, 0, 0, c]
                    else:
                        for kv in range(2):
                            x = x * stats[0, kv, r] + b[0, kv, r, c]
                        expected = x / stats[0, -1, r] if case.endswith("finalize") else x
                    self.assertAlmostEqual(float(actual[0, r, c]), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
