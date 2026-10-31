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

from catalog import CASES, mode_configs, validate
import run_workloads


class ReportTests(unittest.TestCase):
    def test_mode_switch_matrix(self):
        expected = {"baseline": (False, False), "ownership": (True, False),
                    "padding": (False, True), "auto": (True, True)}
        for mode, flags in expected.items():
            config = mode_configs(mode)
            self.assertEqual((config["tl.ascend_enable_ub_ownership"], config["tl.ascend_enable_ub_padding"]), flags)

    def test_source_provenance(self):
        self.assertEqual(sorted(Counter(c.category for c in CASES.values()).values()), [2, 2, 2])
        root = Path(__file__).resolve().parents[3]
        for case in CASES.values():
            tree = ast.parse((root / case.source).read_text())
            functions = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for symbol in case.symbol.split(" + "):
                self.assertIn(symbol.split("(")[0], functions)

    def test_invalid_dimensions(self):
        validate(32, 32, 32, 32, 4)
        for dimensions in [(32, 17, 32, 32, 4), (16, 32, 32, 32, 4), (32, 32, 16, 32, 4), (32, 32, 32, 0, 4)]:
            with self.assertRaises(ValueError):
                validate(*dimensions)

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
            for r in range(3):
                for c in range(4):
                    x = float(a[0, r, c])
                    if case == "gemm_scale":
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
