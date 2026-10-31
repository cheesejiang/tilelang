"""Scan tile shapes at fixed blocks, then blocks at a fixed tile shape."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

from catalog import CASES, MODES, iteration_count, validate
import run_workloads  # noqa: F401 - sets up the shared report helpers' import path.
from layout_diagnostics import COST_FIELDS, STORAGE_FIELDS, FINAL_IR_FIELDS, candidate_rows, final_ir_fields, summary_fields


@dataclass(frozen=True)
class Configuration:
    scan: str
    rows: int
    columns: int
    threads: int
    blocks: int

    @property
    def directory(self):
        return Path(self.scan) / f"r{self.rows}_c{self.columns}_t{self.threads}_b{self.blocks}"


def parse_shape(text):
    try:
        rows, columns = text.lower().split("x")
        result = int(rows), int(columns)
        if min(result) <= 0:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid shape {text!r}; use positive ROWSxCOLUMNS, e.g. 32x34") from None


def configurations(args):
    result = []
    if args.sweep in ("shapes", "both"):
        result.extend(Configuration("shapes", r, c, args.threads, args.blocks) for r, c in args.shapes)
    if args.sweep in ("blocks", "both"):
        result.extend(Configuration("blocks", args.rows, args.columns, args.threads, b) for b in args.block_counts)
    result = list(dict.fromkeys(result))
    for config in result:
        validate(config.rows, config.columns, config.threads, config.blocks, args.updates)
        selected = CASES if getattr(args, "case", "all") == "all" else (args.case,)
        for case in selected:
            validate(config.rows, config.columns, config.threads, config.blocks, args.updates, case=case)
    return result


def collect_configuration(args, config):
    rows, candidates = [], []
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    modes = MODES if args.mode == "all" else (args.mode,)
    for case, spec in cases.items():
        reports = {}
        directory = args.output / config.directory / case
        for mode in modes:
            path = directory / f"{mode}.json"
            if path.exists():
                report = json.loads(path.read_text())
                expected = {"rows": config.rows, "columns": config.columns, "threads": config.threads,
                            "blocks": config.blocks,
                            "iterations": iteration_count(case, config.columns, args.updates)}
                if any(report.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Archived configuration does not match scan plan: {path}")
                reports[mode] = report
            elif not args.summarize_only:
                reports[mode] = {"status": "failed_or_not_run"}
        baseline = reports.get("baseline", {})
        for mode, report in reports.items():
            base = baseline if report.get("experiment_protocol") == baseline.get("experiment_protocol") else {}
            latency = report.get("latency_us")
            before = base.get("latency_us")
            metadata = {**asdict(config), "case": case, "mode": mode,
                        "iterations": iteration_count(case, config.columns, args.updates)}
            elements = config.rows * config.columns * config.blocks
            rows.append({**metadata, **summary_fields(report.get("layout_diagnostics", [])),
                         **final_ir_fields(report, base),
                         "status": report.get("status", "unknown"),
                         "correctness_passed": report.get("correctness_passed", "unknown"),
                         "experiment_protocol": report.get("experiment_protocol", "unknown"),
                         "latency_us": latency if latency is not None else "",
                         "speedup_vs_baseline": before / latency if before and latency else "",
                         "output_elements": elements,
                         "output_melems_per_s": elements / latency if latency else "",
                         "report_path": str(directory / f"{mode}.json")})
            candidates.extend({**metadata, **row} for row in candidate_rows(report.get("layout_diagnostics", [])))
    return rows, candidates


def save_sweep(output, rows, candidates):
    output.mkdir(parents=True, exist_ok=True)
    fields = ["scan", "case", "rows", "columns", "threads", "blocks", "iterations", "mode", "status",
              "correctness_passed", "experiment_protocol", "latency_us", "speedup_vs_baseline",
              "output_elements", "output_melems_per_s", "padding_scheme", "cost_scope",
              *STORAGE_FIELDS, *FINAL_IR_FIELDS, *COST_FIELDS,
              *(f"unpadded_{key}" for key in COST_FIELDS), "unmodeled_costs", "report_path"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    candidate_fields = list(dict.fromkeys(k for row in candidates for k in row))
    with (output / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields or ["scan", "case", "mode"])
        writer.writeheader()
        writer.writerows(candidates)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("scan", "case", "rows", "columns", "threads", "blocks", "iterations"))
        groups.setdefault(key, {})[row["mode"]] = row
    lines = ["# Shape / block scan", "",
             "| scan | case | rows | columns | threads | blocks | iterations | baseline (us) | ownership (us) | padding (us) | auto (us) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, modes in groups.items():
        values = []
        for mode in MODES:
            row = modes.get(mode)
            values.append("—" if row is None else f"{row['latency_us']:.3f}" if row["latency_us"] != "" else row["status"])
        lines.append("| " + " | ".join(map(str, (*key, *values))) + " |")
    columns = ["scan", "case", "rows", "columns", "threads", "blocks", "iterations", "mode", "correctness",
               "padding_scheme", "speedup_vs_baseline", "output_melems_per_s", "estimated_ub_extra_allocation_bytes",
               "final_ir_ub_allocation_bytes", "final_ir_ub_extra_vs_baseline_bytes", *COST_FIELDS, "cost_scope"]
    lines.extend(["", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
    for row in rows:
        verified = row["correctness_passed"]
        values = {**row, "correctness": "passed" if verified is True else "not_passed" if verified is False else "unknown"}
        lines.append("| " + " | ".join(f"{values[k]:.3f}" if isinstance(values.get(k), float) else str(values.get(k, "")) for k in columns) + " |")
    lines.extend(["", "shapes: blocks and threads fixed; blocks: shape and threads fixed.",
                  "Output throughput is blocks*rows*columns / latency_us in Melem/s; not memory bandwidth or operation throughput.",
                  "UB allocation and model costs are per block; do not multiply them by blocks when comparing local conflicts.",
                  "Final IR allocation is not device-measured usage; model cycles exclude unmodeled DMA/synchronization/address costs.",
                  "Compare speedups only against the baseline in the same configuration. Single-mode runs have unknown baseline deltas."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["shapes", "blocks", "both"], default="both")
    parser.add_argument("--shapes", type=parse_shape, nargs="+", default=[(32, 16), (32, 32), (32, 34), (32, 40), (64, 32), (128, 32)])
    parser.add_argument("--rows", type=int, default=32, help="fixed rows for block scan")
    parser.add_argument("--columns", type=int, default=32, help="fixed columns for block scan")
    parser.add_argument("--blocks", type=int, default=32, help="fixed blocks for shape scan")
    parser.add_argument("--block-counts", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--threads", type=int, default=32, help="fixed across the entire scan")
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--mode", choices=["all", *MODES], default="all")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print and save scan plan without starting kernels")
    parser.add_argument("--output", type=Path, default=Path("layout-workload-sweeps"))
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    try:
        plan = configurations(args)
        if args.rounds <= 0:
            raise ValueError("rounds must be positive")
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"case": args.case, "mode": args.mode, "updates": args.updates, "seed": args.seed,
                "bench": args.bench, "rounds": args.rounds, "configurations": [asdict(c) for c in plan]}
    if not args.summarize_only:
        (args.output / "scan_plan.json").write_text(json.dumps(manifest, indent=2))
    for config in plan:
        print(f"{config.scan}: {config.rows}x{config.columns}, threads={config.threads}, blocks={config.blocks}", flush=True)
    jobs = len(plan) * (len(CASES) if args.case == "all" else 1) * (len(MODES) if args.mode == "all" else 1)
    print(f"{len(plan)} configurations, {jobs} case/mode runs; summary: {args.output / 'summary.md'}", flush=True)
    if args.dry_run:
        return
    rows, candidates = [], []
    failures = 0
    if not args.summarize_only:
        save_sweep(args.output, [], [])
    for config in plan:
        if not args.summarize_only:
            cmd = [sys.executable, str(Path(__file__).with_name("run_workloads.py")),
                   "--case", args.case, "--mode", args.mode, "--rows", str(config.rows), "--columns", str(config.columns),
                   "--threads", str(config.threads), "--blocks", str(config.blocks), "--updates", str(args.updates),
                   "--rounds", str(args.rounds), "--seed", str(args.seed), "--output", str(args.output / config.directory)]
            if args.bench:
                cmd.append("--bench")
            failures += int(subprocess.run(cmd, check=False).returncode != 0)
        try:
            new_rows, new_candidates = collect_configuration(args, config)
        except ValueError as error:
            parser.error(str(error))
        rows.extend(new_rows)
        candidates.extend(new_candidates)
        if rows:
            save_sweep(args.output, rows, candidates)
    if not rows:
        parser.error("No matching per-mode JSON found; use the same scan arguments and output directory as the original run")
    print(f"Saved {args.output / 'summary.md'}", flush=True)
    if failures:
        raise SystemExit(f"{failures} configuration(s) failed; available results have been saved")


if __name__ == "__main__":
    main()
