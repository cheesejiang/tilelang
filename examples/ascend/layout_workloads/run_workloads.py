"""Six source-backed stage extracts with four-mode layout diagnostics."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from catalog import CASES, MODES, mode_configs, validate

# Reuse the existing report formats, final-allocation reader and summary writer.
SUPPORT = Path(__file__).resolve().parents[1] / "ub_layout_ablation"
sys.path.insert(0, str(SUPPORT))
from layout_diagnostics import collect_final_ub_allocations, final_ir_fields, plain, summary_fields, candidate_rows
from example_reuse_suite import save_summary


def run_worker(args):
    import torch
    import torch_npu  # noqa: F401
    import tilelang
    from tilelang import tvm
    from tvm import tirx
    from tilelang.instrumentation.session import (
        PassInstrumentationTool, register_pass_instrumentation_tool, unregister_pass_instrumentation_tool,
    )
    from tilelang.profiler import do_bench
    from kernels import make_kernel
    from reference import reference

    source = CASES[args.case]
    directory = args.output / args.case
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.mode}.json"
    report = {**asdict(source), "case": args.case, "mode": args.mode,
              "rows": args.rows, "columns": args.columns, "threads": args.threads, "blocks": args.blocks,
              "iterations": args.updates if source.category == "cross_reused" else 1,
              "status": "compiling", "padding_applied": False}
    source_path = Path(__file__).resolve().parents[3] / source.source
    report["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    report["experiment_protocol"] = "identical-ir-pass-switches-v1"
    report["pass_configs"] = mode_configs(args.mode)
    report["baseline_partition"] = "common native layout inference; no mode-specific annotations"
    report["scope"] = "SIMT stage extract, not an end-to-end benchmark of the source kernel"

    def save():
        path.write_text(json.dumps(report, indent=2))

    save()
    try:
        tilelang.disable_cache()
        torch.manual_seed(args.seed)
        a = torch.randn(args.blocks, args.rows, args.columns)
        tiles = args.updates if source.category == "cross_reused" else 1
        b = torch.randn(args.blocks, tiles, args.rows, args.columns)
        stats = torch.rand(args.blocks, tiles + 1, args.rows)
        stats[:, :-1, :] = stats[:, :-1, :] * 0.4 + 0.5
        stats[:, -1, :] = stats[:, -1, :] + 1
        expected = reference(args.case, a, b, stats)
        inputs = [x.to("npu") for x in (a, b, stats)]
        snapshots = {}

        @tvm.ir.instrument.pass_instrument
        class Capture:
            def run_after_pass(self, mod, info):
                if info.name in ("tl.AscendLayoutInference", "tl.MergeUBAllocations"):
                    snapshots[info.name] = mod

        class CaptureTool(PassInstrumentationTool):
            def create_pass_instrument(self):
                return Capture()

        name = "examples.layout_workloads.capture"
        register_pass_instrumentation_tool(name, CaptureTool)
        try:
            func = make_kernel(args.case, args.rows, args.columns, args.threads, args.blocks, tiles)
            report["input_ir_structural_hash"] = str(tvm.ir.structural_hash(func, map_free_vars=True))
            (directory / f"{args.mode}.input.tir").write_text(str(func))
            compiled = tilelang.compile(func, target="ascend", out_idx=-1,
                                        pass_configs=report["pass_configs"])
        finally:
            unregister_pass_instrumentation_tool(name)
        (directory / f"{args.mode}.cpp").write_text(compiled.get_kernel_source())
        records = []
        loops = []
        inferred = snapshots.get("tl.AscendLayoutInference")
        if inferred is None:
            raise RuntimeError("Missing AscendLayoutInference snapshot; ensure caching is disabled")
        (directory / f"{args.mode}.tir").write_text(str(inferred))
        for func in inferred.functions.values():
            if not isinstance(func, tirx.PrimFunc):
                continue
            if func.attrs is not None and "tl.ascend_layout_diagnostics" in func.attrs:
                records.extend(plain(func.attrs["tl.ascend_layout_diagnostics"]))

            def inspect(node):
                if isinstance(node, tirx.For) and "parallel_loop_layout" in node.annotations:
                    loops.append(str(node.annotations["parallel_loop_layout"]))

            tirx.stmt_functor.post_order_visit(func.body, inspect)
        if not records:
            raise RuntimeError("Missing layout diagnostics; rebuild the TileLang library")
        report["layout_diagnostics"] = records
        report["loop_layouts"] = loops
        report.update(summary_fields(records))
        report["padding_applied"] = any("padding_bytes" in b for r in records if r.get("stage") == "final" for b in r.get("buffers", []))
        report["padding_policy"] = "rule_based_search" if report["pass_configs"]["tl.ascend_enable_ub_padding"] else "disabled"
        allocated = snapshots.get("tl.MergeUBAllocations")
        if allocated is not None:
            report["final_ir_ub"] = collect_final_ub_allocations(allocated)
            (directory / f"{args.mode}.allocated.tir").write_text(str(allocated))
        report.update(final_ir_fields(report))
        report["status"] = "compiled"
        save()
        if not report["pass_configs"]["tl.ascend_enable_ub_padding"] and report["padding_applied"]:
            raise RuntimeError("Padding was applied despite its optimization switch being disabled")
        actual = compiled(*inputs)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
        report["correctness_passed"] = True
        report["status"] = "correctness_passed"
        save()
        if args.bench:
            samples = [do_bench(lambda: compiled(*inputs), backend="msprof", _n_warmup=10, _n_repeat=30) * 1000
                       for _ in range(args.rounds)]
            report.update(latency_us=statistics.median(samples), latency_samples_us=samples, status="measured")
            save()
        print(f"{args.case}/{args.mode}: {report['status']}; padding={report['padding_applied']}; "
              f"latency_us={report.get('latency_us', 'not_measured')}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


def summarize_reports(args, cases, reports_by_case):
    records, candidates = [], []
    for case, reports in reports_by_case.items():
        baseline = reports.get("baseline", {})
        for mode, report in reports.items():
            # Rebuilding a table must use the archived configuration, not
            # today's CLI defaults. Do not compare incompatible old runs.
            compatible = all(report.get(k) == baseline.get(k) for k in
                             ("rows", "columns", "threads", "blocks", "iterations", "experiment_protocol"))
            base = baseline if compatible else {}
            latency = report.get("latency_us", "")
            base_latency = base.get("latency_us")
            config = {"case": case, "columns": report.get("columns", args.columns), "row_stride": 1,
                      "iterations": report.get("iterations", args.updates if cases[case].category == "cross_reused" else 1),
                      "blocks": report.get("blocks", args.blocks), "mode": mode}
            candidates.extend({**config, **row} for row in candidate_rows(report.get("layout_diagnostics", [])))
            final = final_ir_fields(report, base)
            path = args.output / case / f"{mode}.json"
            if not args.summarize_only and path.exists():
                report.update(final)
                path.write_text(json.dumps(report, indent=2))
            records.append({**config, **summary_fields(report.get("layout_diagnostics", [])), **final,
                            "status": report.get("status", "failed"), "padding_applied": report.get("padding_applied", ""),
                            "latency_us": latency, "speedup_vs_baseline": base_latency / latency if base_latency and latency else "",
                            "report_path": str(path)})
    save_summary(args.output, records, candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--mode", choices=["all", *MODES], default="all")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--updates", type=int, default=4, help="number of different incoming KV tiles, not dummy repetitions")
    parser.add_argument("--rounds", type=int, default=3, help="independent benchmark batches; report their median")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="rebuild tables from existing JSON without compiling or running kernels")
    parser.add_argument("--output", type=Path, default=Path("layout-workload-results"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    try:
        validate(args.rows, args.columns, args.threads, args.blocks, args.updates)
        if args.rounds <= 0:
            raise ValueError("rounds must be positive")
    except ValueError as error:
        parser.error(str(error))
    if args.worker:
        if args.summarize_only:
            parser.error("--worker cannot be combined with --summarize-only")
        if args.case == "all" or args.mode == "all":
            parser.error("worker requires one case and mode")
        run_worker(args)
        return
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    modes = MODES if args.mode == "all" else (args.mode,)
    reports_by_case = {}
    failures = 0
    print(f"Summary path: {args.output / 'summary.md'}", flush=True)
    if not args.summarize_only:
        save_summary(args.output, [], [])
    for case, spec in cases.items():
        reports = reports_by_case.setdefault(case, {})
        for mode in modes:
            directory = args.output / case
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{mode}.json"
            if args.summarize_only:
                if not path.exists():
                    continue
            else:
                path.unlink(missing_ok=True)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", case, "--mode", mode,
                       "--rows", str(args.rows), "--columns", str(args.columns), "--threads", str(args.threads),
                       "--blocks", str(args.blocks), "--updates", str(args.updates), "--rounds", str(args.rounds),
                       "--seed", str(args.seed), "--output", str(args.output)]
                if args.bench:
                    cmd.append("--bench")
                result = subprocess.run(cmd, check=False)
                failures += int(result.returncode != 0)
            reports[mode] = json.loads(path.read_text()) if path.exists() else {"status": "failed"}
            summarize_reports(args, cases, reports_by_case)
            print(f"Saved {args.output / 'summary.md'} ({case}/{mode})", flush=True)
    if not any(reports_by_case.values()):
        parser.error(f"No per-mode JSON files found under {args.output}")
    (args.output / "cases.json").write_text(json.dumps({name: asdict(case) for name, case in cases.items()}, indent=2))
    print((args.output / "summary.md").read_text())
    if failures:
        raise SystemExit(f"{failures} worker(s) failed; see per-mode JSON and captured tracebacks")


if __name__ == "__main__":
    main()
