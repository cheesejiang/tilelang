"""Study planning, rectangular transpose references and trustworthy comparisons."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from catalog import validate
import run_padding_study as study


def arguments(**overrides):
    return SimpleNamespace(**({"study": "both", "short_scan": "both", "rows": 128,
                              "short_cases": ["gemm_scale", "attention_finalize"],
                              "threads": 32, "blocks": 32, "updates": 4, "rounds": 3,
                              "seed": 0, "bench": True, "no_manual_references": False,
                              "output": Path("unused")} | overrides))


class PaddingStudyTests(unittest.TestCase):
    def test_plan_covers_both_directions_and_manual_controls(self):
        jobs = study.plan(arguments())
        self.assertEqual(len(jobs), 120)
        equal = [j for j in jobs if j.scan == "equal-elements"]
        self.assertEqual({j.rows * j.columns for j in equal}, {4096})
        fixed = [j for j in jobs if j.scan == "fixed-rows"]
        self.assertEqual({j.rows for j in fixed}, {128})
        transpose = [j for j in jobs if j.case == "transpose"]
        self.assertEqual(len(transpose), 24)
        self.assertEqual({j.transpose_layout for j in transpose}, {"native", "row32", "source4_repack"})
        for job in transpose:
            if job.transpose_layout != "native":
                self.assertEqual(job.mode, "baseline")
        self.assertEqual(len({(j.directory, j.case, j.mode) for j in jobs}), len(jobs))

    def test_reference_validation_accounts_for_extra_repack_storage(self):
        validate(64, 128, 128, 1, 4, case="transpose")
        validate(128, 128, 32, 1, 4, case="transpose", transpose_layout="source4_repack")
        with self.assertRaisesRegex(ValueError, "reference exceeds"):
            validate(128, 160, 32, 1, 4, case="transpose", transpose_layout="source4_repack")
        with self.assertRaisesRegex(ValueError, "require --case transpose"):
            validate(32, 32, 32, 1, 4, case="gemm_scale", transpose_layout="row32")

    def test_reports_distinguish_manual_ir_and_reject_incompatible_native_ir(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(output=Path(folder))
            jobs = [study.Job("transpose", "tiles", "transpose", 64, 128, 32, 32, mode, layout)
                    for mode, layout in (("baseline", "native"), ("padding", "native"),
                                         ("auto", "native"), ("baseline", "source4_repack"))]
            for job in jobs:
                path = args.output / job.directory / job.case / f"{job.mode}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                report = {key: getattr(job, key) for key in
                          ("case", "mode", "rows", "columns", "threads", "blocks", "transpose_layout")}
                report.update(iterations=1, seed=0, kernel_factory_sha256="factory", status="measured",
                              correctness_passed=True, experiment_protocol="identical-ir-pass-switches-v1",
                              input_ir_structural_hash="changed" if job.mode == "auto" else "same",
                              latency_us=10 if job.mode == "baseline" and job.transpose_layout == "native" else 8,
                              layout_diagnostics=[])
                if job.transpose_layout != "native":
                    report.update(manual_padding_scheme="+4 float32 with repack", input_ir_structural_hash="manual",
                                  experiment_protocol="manual-transpose-source4_repack-v1")
                path.write_text(json.dumps(report))
            rows = study.collect(args, jobs)
            self.assertEqual(rows[1]["speedup_vs_native_baseline"], 1.25)
            self.assertEqual(rows[2]["speedup_vs_native_baseline"], "")
            self.assertEqual(rows[3]["input_ir_matches_baseline"], "manual_reference")
            self.assertEqual(rows[3]["speedup_vs_native_baseline"], 1.25)
            study.save(args, rows)
            with (args.output / "summary.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            bad = args.output / jobs[0].directory / "transpose/baseline.json"
            report = json.loads(bad.read_text())
            report["columns"] = 32
            bad.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "columns"):
                study.collect(args, jobs)

    def test_dry_run_does_not_import_device_modules_or_spawn(self):
        with tempfile.TemporaryDirectory() as folder:
            argv = ["run_padding_study.py", "--study", "transpose", "--dry-run", "--output", folder]
            with patch("sys.argv", argv), patch.object(study.subprocess, "run") as child, redirect_stdout(io.StringIO()):
                study.main()
            child.assert_not_called()
            self.assertEqual(len(json.loads((Path(folder) / "study_plan.json").read_text())["jobs"]), 24)


if __name__ == "__main__":
    unittest.main()
