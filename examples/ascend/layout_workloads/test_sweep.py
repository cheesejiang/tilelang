"""Host-only checks of scan isolation and cross-configuration report recovery."""

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sweep_workloads as sweep


class SweepTests(unittest.TestCase):
    def test_shape_parser(self):
        self.assertEqual(sweep.parse_shape("32X34"), (32, 34))
        for shape in ("32", "32x0", "32x32x32", "axb"):
            with self.assertRaises(argparse.ArgumentTypeError):
                sweep.parse_shape(shape)

    def test_scans_change_only_one_factor(self):
        args = SimpleNamespace(sweep="both", shapes=[(32, 16), (64, 34), (32, 16)],
                               rows=32, columns=40, threads=32, blocks=16, block_counts=[1, 8], updates=4)
        plan = sweep.configurations(args)
        self.assertEqual(len(plan), 4)
        shapes = [c for c in plan if c.scan == "shapes"]
        self.assertEqual({c.blocks for c in shapes}, {16})
        blocks = [c for c in plan if c.scan == "blocks"]
        self.assertEqual({(c.rows, c.columns) for c in blocks}, {(32, 40)})
        self.assertEqual({c.threads for c in plan}, {32})
        self.assertEqual(len({c.directory for c in plan}), 4)

    def test_scan_aggregates_shapes_and_recovers_without_device(self):
        with tempfile.TemporaryDirectory() as folder:
            def child(cmd, check):
                directory = Path(cmd[cmd.index("--output") + 1]) / "gemm_scale"
                directory.mkdir(parents=True)
                config = {k: int(cmd[cmd.index("--" + k) + 1]) for k in ("rows", "columns", "threads", "blocks")}
                for mode in ("baseline", "ownership", "padding", "auto"):
                    report = {**config, "iterations": 1, "status": "measured", "correctness_passed": True,
                              "experiment_protocol": "identical-ir-pass-switches-v1",
                              "latency_us": 8.0 if mode == "auto" else 10.0,
                              "final_ir_ub": {"status": "known", "ub_allocation_bytes": 4192 if mode == "auto" else 4096},
                              "layout_diagnostics": [{"stage": "final", "entries": 64 if mode == "auto" else 2048}]}
                    (directory / f"{mode}.json").write_text(json.dumps(report))
                return SimpleNamespace(returncode=0)

            argv = ["sweep_workloads.py", "--sweep", "shapes", "--shapes", "32x16", "32x34",
                    "--case", "gemm_scale", "--bench", "--output", folder]
            with patch("sys.argv", argv), patch("sweep_workloads.subprocess.run", side_effect=child), redirect_stdout(io.StringIO()):
                sweep.main()
            path = Path(folder) / "summary.csv"
            first = path.read_text()
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["columns"] for r in rows}, {"16", "34"})
            for row in rows:
                self.assertEqual(row["blocks"], "32")
                if row["mode"] == "auto":
                    self.assertEqual(float(row["speedup_vs_baseline"]), 1.25)
                    self.assertEqual(row["final_ir_ub_extra_vs_baseline_bytes"], "96")
                    expected = int(row["rows"]) * int(row["columns"]) * 32 / 8
                    self.assertEqual(float(row["output_melems_per_s"]), expected)
            with patch("sys.argv", argv + ["--summarize-only"]), patch("sweep_workloads.subprocess.run") as child_process, redirect_stdout(io.StringIO()):
                sweep.main()
            child_process.assert_not_called()
            self.assertEqual(path.read_text(), first)
            text = (Path(folder) / "summary.md").read_text()
            self.assertIn("| shapes | gemm_scale | 32 | 34 |", text)


if __name__ == "__main__":
    unittest.main()
