"""Complete transpose and short-axis studies; host-only planning/report recovery."""
import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

from catalog import MODES, iteration_count, validate


@dataclass(frozen=True)
class Job:
    study: str
    scan: str
    case: str
    rows: int
    columns: int
    threads: int
    blocks: int
    mode: str
    transpose_layout: str = "native"

    @property
    def directory(self):
        return (Path(self.study) / self.scan /
                f"r{self.rows}_c{self.columns}_t{self.threads}_b{self.blocks}" /
                self.transpose_layout)

    @property
    def key(self):
        return self.study, self.scan, self.case, self.rows, self.columns, self.threads, self.blocks


def plan(args):
    jobs = []
    if args.study in ("both", "short-axis"):
        shapes = []
        if args.short_scan in ("both", "fixed-rows"):
            shapes += [("fixed-rows", args.rows, c) for c in (4, 8, 16, 32, 34, 64)]
        if args.short_scan in ("both", "equal-elements"):
            # Each tile has 4096 elements; axis length changes, traffic does not.
            shapes += [("equal-elements", 4096 // c, c) for c in (4, 8, 16, 32, 64, 128)]
        for scan, rows, columns in shapes:
            for case in args.short_cases:
                for mode in MODES:
                    jobs.append(Job("short-axis", scan, case, rows, columns,
                                    args.threads, args.blocks, mode))
    if args.study in ("both", "transpose"):
        for rows, columns in ((64, 64), (64, 128), (128, 64), (128, 128)):
            for mode in MODES:
                jobs.append(Job("transpose", "tiles", "transpose", rows, columns,
                                args.threads, args.blocks, mode))
            if not args.no_manual_references:
                for storage in ("row32", "source4_repack"):
                    jobs.append(Job("transpose", "tiles", "transpose", rows, columns,
                                    args.threads, args.blocks, "baseline", storage))
    for job in jobs:
        validate(job.rows, job.columns, job.threads, job.blocks, args.updates,
                 case=job.case, transpose_layout=job.transpose_layout)
    return list(dict.fromkeys(jobs))


def command(args, job):
    result = [sys.executable, str(Path(__file__).with_name("run_workloads.py")),
              "--case", job.case, "--mode", job.mode,
              "--rows", str(job.rows), "--columns", str(job.columns),
              "--threads", str(job.threads), "--blocks", str(job.blocks),
              "--updates", str(args.updates), "--rounds", str(args.rounds),
              "--seed", str(args.seed), "--transpose-layout", job.transpose_layout,
              "--output", str(args.output / job.directory)]
    if args.bench:
        result.append("--bench")
    return result


def collect(args, jobs):
    # Import only when reporting: the shared helper needs no TVM/NPU.
    import run_workloads  # noqa: F401 -- locates shared diagnostics helpers
    from layout_diagnostics import final_ir_fields, summary_fields

    loaded = []
    for job in jobs:
        path = args.output / job.directory / job.case / f"{job.mode}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        expected = {key: getattr(job, key) for key in
                    ("case", "mode", "rows", "columns", "threads", "blocks", "transpose_layout")}
        expected["iterations"] = iteration_count(job.case, job.columns, args.updates)
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f"Archived {key} does not match study plan: {path}")
        loaded.append((job, report, path))
    baselines = {j.key: r for j, r, _ in loaded if j.mode == "baseline" and j.transpose_layout == "native"}
    rows = []
    for job, report, path in loaded:
        baseline = baselines.get(job.key, {})
        # Manual references deliberately have different IR. Compare timing only,
        # keeping the extra repack in the measured reference kernel.
        matched = bool(baseline) and all(report.get(key) == baseline.get(key) for key in
                                        ("seed", "kernel_factory_sha256"))
        if job.transpose_layout == "native":
            matched &= all(report.get(key) == baseline.get(key) for key in
                           ("experiment_protocol", "input_ir_structural_hash"))
            matched &= bool(report.get("input_ir_structural_hash"))
        matched &= bool(report.get("kernel_factory_sha256"))
        def measured(r):
            value = r.get("latency_us")
            return (r.get("status") == "measured" and r.get("correctness_passed") is True
                    and isinstance(value, (int, float)) and value > 0)
        speedup = (baseline["latency_us"] / report["latency_us"]
                   if matched and measured(baseline) and measured(report) else "")
        diagnostic = summary_fields(report.get("layout_diagnostics", []))
        scheme = report.get("manual_padding_scheme", diagnostic.get("padding_scheme", "unknown"))
        audits = report.get("ub_store_audits", {})
        audit = audits.get("padded" if job.transpose_layout == "source4_repack" else
                           "transposed" if job.case == "transpose" else "work", {})
        final = final_ir_fields(report, baseline if matched else {})
        rows.append({**asdict(job), "iterations": iteration_count(job.case, job.columns, args.updates),
                     "status": report.get("status", "unknown"),
                     "correctness": "passed" if report.get("correctness_passed") is True else "unknown",
                     "padding_applied": report.get("padding_applied", "unknown"),
                     "padding_scheme": scheme, "latency_us": report.get("latency_us", ""),
                     "speedup_vs_native_baseline": speedup,
                     "input_ir_matches_baseline": (matched if job.transpose_layout == "native" else "manual_reference"),
                     "estimated_latency_cycles": diagnostic.get("estimated_latency_cycles", "unknown"),
                     "entries": diagnostic.get("entries", "unknown"),
                     "cost_scope": diagnostic.get("cost_scope", "unknown"),
                     "final_ir_ub_allocation_bytes": final.get("final_ir_ub_allocation_bytes", "unknown"),
                     "final_ir_ub_extra_vs_baseline_bytes": final.get("final_ir_ub_extra_vs_baseline_bytes", "unknown"),
                     "audit_status": audit.get("status", "unknown"),
                     "max_active_lanes": audit.get("max_active_lanes", "unknown"),
                     "max_distinct_lines_per_group": audit.get("max_distinct_lines_per_group", "unknown"),
                     "dma_descriptors": json.dumps(report.get("dma_audit", {}).get("descriptors", [])),
                     "report_path": str(path)})
    return rows


def save(args, rows):
    if not rows:
        return
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ("study", "scan", "case", "rows", "columns", "mode", "transpose_layout",
               "correctness", "padding_scheme", "latency_us", "speedup_vs_native_baseline",
               "entries", "final_ir_ub_allocation_bytes", "max_active_lanes", "audit_status")
    lines = ["# Padding studies", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.4f}" if isinstance(row[k], float)
                                       else str(row[k]) for k in columns) + " |")
    lines += ["", "All automatic modes use identical input IR; model cycles are not measured kernel time.",
              "Manual transpose references have different IR; source4_repack includes its dense UB repack and extra storage.",
              "The +4-element reference does not reproduce the source CUDA microtile/rotation schedule.",
              "No padding selected is a valid outcome, not evidence of a padding speedup.",
              "Store audits describe scalar lowered TIR, not backend ISA; vectorized or unresolved audits remain unknown.",
              "Equal-elements scans keep useful tile elements fixed; fixed-rows scans change useful traffic.",
              "Inspect each report's diagnostics, DMA descriptors and saved TIR/C++ before attributing timing to bank conflicts."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["both", "short-axis", "transpose"], default="both")
    parser.add_argument("--short-scan", choices=["both", "fixed-rows", "equal-elements"], default="both")
    parser.add_argument("--short-cases", nargs="+", choices=["gemm_scale", "attention_finalize", "kda_column_store"],
                        default=["gemm_scale", "attention_finalize"])
    parser.add_argument("--rows", type=int, default=128, help="rows for fixed-rows short-axis scan")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--no-manual-references", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("padding-study-results"))
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    try:
        if args.rounds <= 0:
            raise ValueError("rounds must be positive")
        if args.dry_run and args.summarize_only:
            raise ValueError("--dry-run and --summarize-only cannot be combined")
        jobs = plan(args)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"jobs": [asdict(j) for j in jobs], "updates": args.updates,
                "seed": args.seed, "bench": args.bench, "rounds": args.rounds}
    path = args.output / "study_plan.json"
    if args.summarize_only:
        if not path.exists() or json.loads(path.read_text()) != manifest:
            parser.error("Use the original study arguments (including --bench) to recover these reports")
        rows = collect(args, jobs)
        if not rows:
            parser.error("No matching reports found")
        save(args, rows)
        return
    path.write_text(json.dumps(manifest, indent=2))
    print(f"{len(jobs)} case/mode jobs; plan: {path}", flush=True)
    for job in jobs:
        print(f"{job.study}/{job.scan}: {job.case} {job.rows}x{job.columns} "
              f"{job.mode}/{job.transpose_layout}", flush=True)
    if args.dry_run:
        return
    failures = 0
    # Only reports produced during this invocation are considered, so stale
    # JSON cannot masquerade as a completed measurement after a failed child.
    completed = []
    for job in jobs:
        report = args.output / job.directory / job.case / f"{job.mode}.json"
        report.unlink(missing_ok=True)
        log = args.output / job.directory / job.case / f"{job.mode}.worker.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            status = subprocess.run(command(args, job), stdout=stream, stderr=subprocess.STDOUT,
                                    check=False).returncode
        failures += int(status != 0)
        completed.append(job)
        save(args, collect(args, completed))
        print(f"{job.case}/{job.mode}/{job.transpose_layout}: "
              f"{'failed' if status else 'finished'}; log={log}", flush=True)
    if failures:
        raise SystemExit(f"{failures} job(s) failed; available reports retained under {args.output}")
    print(f"Summary: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
