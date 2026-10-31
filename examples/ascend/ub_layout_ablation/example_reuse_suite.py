"""Sweep direct DMA kernels; each configuration runs in a fresh process."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from layout_diagnostics import COST_FIELDS, STORAGE_FIELDS, FINAL_IR_FIELDS, candidate_rows, final_ir_fields, summary_fields


CASES = {
    "row128": ("reuse", 32, 1),
    "row64": ("reuse", 16, 1),
    "stride16": ("reuse", 32, 16),
    "rotate": ("rotate", 32, 1),
}


MODES = ("baseline", "ownership", "padding", "auto")


def save_summary(output, records, candidates=()):
    """Persist after every configuration so a later failure loses no timings."""
    output.mkdir(parents=True, exist_ok=True)
    fields = ["case", "columns", "row_stride", "iterations", "blocks", "mode",
              "status", "correctness_passed", "padding_applied", "latency_us", "speedup_vs_baseline",
              "padding_scheme", *STORAGE_FIELDS, *FINAL_IR_FIELDS, *COST_FIELDS,
              *(f"unpadded_{key}" for key in COST_FIELDS), "unmodeled_costs", "report_path"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    candidate_fields = list(dict.fromkeys(key for row in candidates for key in row))
    with (output / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields or ["case", "mode", "status"])
        writer.writeheader()
        writer.writerows(candidates)
    groups = {}
    for row in records:
        key = (row["case"], row["columns"], row["row_stride"], row["iterations"], row["blocks"])
        groups.setdefault(key, {})[row["mode"]] = row
    lines = ["| case | columns | row stride | iterations | blocks | baseline (us) | ownership (us) | padding (us) | auto (us) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, modes in groups.items():
        values = []
        for mode in MODES:
            row = modes.get(mode)
            if row is None:
                values.append("—")
            elif row["latency_us"] != "":
                values.append(f"{row['latency_us']:.3f}")
            else:
                values.append(row["status"])
        lines.append("| " + " | ".join(map(str, (*key, *values))) + " |")
    lines.extend(["", "| case | iterations | mode | correctness | padding scheme | extra UB (B) | estimated extra allocation (B) | final IR UB allocation (B) | final IR extra vs baseline (B) | UB latency (cycles) | service (cycles) | instructions | requests | entries | partial |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in records:
        keys = ("case", "iterations", "mode", "padding_scheme", "ub_extra_storage_bytes", "estimated_ub_extra_allocation_bytes",
                "final_ir_ub_allocation_bytes", "final_ir_ub_extra_vs_baseline_bytes", *COST_FIELDS)
        verified = row.get("correctness_passed")
        correctness = "passed" if verified is True else "not_passed" if verified is False else "unknown"
        values = [str(row.get(key, "")) for key in keys]
        values.insert(3, correctness)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Estimated allocation values sum per-buffer sizes rounded to 32B without alias reuse; they are not final UB allocation or device measurements.",
                  "Final IR allocation is read after MergeUBAllocations (per-function UB totals, module maximum); it is not a device compiler report or runtime measurement. Unknown baseline deltas are not zero.",
                  "Costs cover modeled SIMT UB traffic per block; DMA, address computation and synchronization are not modeled.",
                  "Unpadded costs retain final ownership and hypothetically remove UB storage layouts; they are not the original ownership cost."])
    table = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--mode", choices=["all", "baseline", "ownership", "padding", "auto"], default="all")
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ub-reuse-results"))
    args = parser.parse_args()
    if args.blocks <= 0 or any(n <= 0 for n in args.iterations):
        parser.error("blocks and iterations must be positive")
    script = Path(__file__).with_name("example_dma_padding.py")
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    records = []
    candidates = []
    modes = MODES if args.mode == "all" else (args.mode,)
    for name, (case, columns, stride) in cases.items():
        for iterations in args.iterations:
            print(f"\n{name}: iterations={iterations}", flush=True)
            directory = args.output / name / str(iterations)
            directory.mkdir(parents=True, exist_ok=True)
            # A failed run must not reuse timings left by a previous invocation.
            for mode in modes:
                (directory / f"{mode}.json").unlink(missing_ok=True)
            cmd = [sys.executable, str(script), "--case", case, "--columns", str(columns),
                   "--row-stride", str(stride), "--iterations", str(iterations),
                   "--blocks", str(args.blocks), "--mode", args.mode,
                   "--output", str(args.output / name / str(iterations))]
            if args.bench:
                cmd.append("--bench")
            completed = subprocess.run(cmd, check=False)
            reports = {}
            for mode in modes:
                path = directory / f"{mode}.json"
                reports[mode] = json.loads(path.read_text()) if path.exists() else {}
            baseline = reports.get("baseline", {}).get("latency_us")
            for mode, report in reports.items():
                latency = report.get("latency_us")
                status = ("measured" if latency is not None else
                          "correctness_passed" if report.get("correctness_passed") else
                          "failed_or_not_run" if completed.returncode else "not_measured")
                configuration = {"case": name, "columns": columns, "row_stride": stride, "iterations": iterations, "blocks": args.blocks, "mode": mode}
                candidates.extend({**configuration, **row} for row in candidate_rows(report.get("layout_diagnostics", [])))
                records.append({**summary_fields(report.get("layout_diagnostics", [])),
                                **final_ir_fields(report, reports.get("baseline")),
                                "report_path": str(directory / f"{mode}.json"),"case": name, "columns": columns, "row_stride": stride,
                                "iterations": iterations, "blocks": args.blocks, "mode": mode,
                                "status": status, "correctness_passed": report.get("correctness_passed", "unknown"),
                                "padding_applied": report.get("padding_applied", ""),
                                "latency_us": latency if latency is not None else "",
                                "speedup_vs_baseline": baseline / latency if baseline and latency else ""})
            table = save_summary(args.output, records, candidates)
            if completed.returncode:
                print(table, flush=True)
                raise SystemExit(completed.returncode)
    print("\n" + table, flush=True)
    print(f"Saved {args.output / 'summary.csv'} and {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
