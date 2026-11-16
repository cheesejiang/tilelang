"""Host-only checks for durable results and worker argument forwarding."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare


class DriverTests(unittest.TestCase):
    def run_driver(self, folder, callback):
        argv = ["compare.py", "--cases", "vecadd_simt", "--profile", "full",
                "--samples", "3", "--repeats", "5", "--output", str(folder)]
        def run(command, **kwargs):
            if "--worker" not in command:
                return SimpleNamespace(returncode=0, stdout="host-test", stderr="")
            return callback(command)
        with patch.object(sys, "argv", argv), patch.object(compare.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            compare.main()

    def test_fresh_result_and_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "vecadd_simt.json").write_text(json.dumps({"status": "passed", "speedup": 999}))
            def callback(command):
                self.assertEqual(command[command.index("--profile") + 1], "full")
                self.assertEqual(command[command.index("--samples") + 1], "3")
                self.assertEqual(command[command.index("--repeats") + 1], "5")
                destination = Path(command[command.index("--output") + 1])
                self.assertNotEqual(destination, output)
                (destination / "vecadd_simt.json").write_text(json.dumps(
                    dict(case="vecadd_simt", status="passed", comparison_type="eager_standard", speedup=1.25)))
                return SimpleNamespace(returncode=0)
            self.run_driver(output, callback)
            self.assertEqual(json.loads((output / "vecadd_simt.json").read_text())["speedup"], 1.25)
            with (output / "results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["speedup"], "1.25")

    def test_crash_cannot_reuse_old_success(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "vecadd_simt.json").write_text(json.dumps({"status": "passed", "speedup": 999}))
            with self.assertRaises(SystemExit) as caught:
                self.run_driver(output, lambda command: SimpleNamespace(returncode=-6))
            self.assertEqual(caught.exception.code, 1)
            row = json.loads((output / "vecadd_simt.json").read_text())
            self.assertEqual(row["status"], "worker_crashed")
            self.assertNotIn("speedup", row)

    def test_timeout_is_saved_without_speedup(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            def callback(command):
                raise subprocess.TimeoutExpired(command, 900)
            with self.assertRaises(SystemExit):
                self.run_driver(output, callback)
            row = json.loads((output / "vecadd_simt.json").read_text())
            self.assertEqual(row["status"], "timeout")
            self.assertNotIn("speedup", row)


if __name__ == "__main__":
    unittest.main()
