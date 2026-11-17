"""Host-only checks for whole-suite compiler performance comparison."""
import tempfile
from pathlib import Path
import unittest
import csv
import contextlib
import io
import sys
from types import SimpleNamespace
from unittest.mock import patch

from compiler_blocks import summarize, write_markdown
import compiler_blocks


class CompilerBlocksTests(unittest.TestCase):
    def test_default_job_matrix_covers_all_cases_both_profiles_and_switches(self):
        seen = set()
        with tempfile.TemporaryDirectory() as directory:
            def run(command, **kwargs):
                case = command[command.index("--cases") + 1]
                profile = command[command.index("--profile") + 1]
                mode = command[command.index("--compiler-auto-blocks") + 1]
                seen.add((case, profile, mode))
                path = Path(command[command.index("--output") + 1]) / "results.csv"
                row = dict(case=case, profile=profile, status="passed", compiler_auto_blocks=mode,
                           device_name="9579", shape="same", dtype="bf16", timing="profiler", arch="dav-3510",
                           tilelang_latency_us=10, speedup=1, compiler_changed="no", blocks=32)
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                return SimpleNamespace(returncode=0)
            with patch.object(sys, "argv", ["compiler_blocks.py", "--rounds", "1", "--output", directory]), \
                    patch.object(compiler_blocks.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                compiler_blocks.main()
            self.assertTrue((Path(directory) / "summary.md").exists())
        self.assertEqual(seen, {(case, profile, mode) for case in compiler_blocks.CASES
                                for profile in ("smoke", "full") for mode in ("off", "on")})

    def test_gain_is_own_baseline_and_no_rewrite_is_still_measured(self):
        common = dict(case="gqa", profile="smoke", device_name="9579", shape="same", dtype="bf16",
                      arch="dav-3510", timing="profiler", status="passed", blocks="32")
        rows = [dict(common, compiler_auto_blocks="off", tilelang_latency_us="20", speedup="2"),
                dict(common, compiler_auto_blocks="on", tilelang_latency_us="10", speedup="3", compiler_changed="no")]
        summary = summarize(rows)
        self.assertEqual(summary[0]["gain"], 2)
        self.assertEqual(summary[0]["after_s"], 3)
        self.assertEqual(summary[0]["compiler_changed"], "no")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            write_markdown(path, summary)
            self.assertIn("20.000 | 10.000 | 2.000× | 2.000× | 3.000×", path.read_text())
        rows[0]["status"] = "failed"
        self.assertNotIn("gain", summarize(rows)[0])
        rows[0]["status"] = "passed"
        rows[0]["device_name"] = "other"
        self.assertNotIn("gain", summarize(rows)[0])

    def test_missing_side_is_not_a_pass(self):
        row = dict(case="vecadd_simt", profile="full", compiler_auto_blocks="on", status="passed")
        self.assertEqual(summarize([row])[0]["status"], "incomplete_or_failed")


if __name__ == "__main__":
    unittest.main()
