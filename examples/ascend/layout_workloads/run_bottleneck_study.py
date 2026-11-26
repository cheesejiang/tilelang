"""Controlled real gemm_scale runs: ownership, padding periods, vectorization."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def variants():
    return [("baseline", "baseline", 0, "default"),
            ("ownership", "ownership", 0, "default"),
            ("padding-search", "padding", 0, "default"),
            ("baseline-scalar", "baseline", 0, "disable"),
            ("p8-scalar", "baseline", 8, "disable"),
            *[(f"p{p}", "baseline", p, "default") for p in (1, 2, 4, 8, 16)]]


def command(args, variant):
    name, mode, period, vectorize = variant
    cmd = [sys.executable, str(Path(__file__).with_name("run_workloads.py")),
           "--case", "gemm_scale", "--mode", mode, "--rows", "32", "--columns", "128",
           "--threads", "32", "--blocks", str(args.blocks), "--rounds", str(args.rounds),
           "--bench", "--vectorize", vectorize, "--scale-padding-period", str(period),
           "--output", str(args.output / name)]
    if not args.no_profile:
        cmd += ["--profile", "--profile-repeats", str(args.profile_repeats)]
    return cmd


def summarize(args):
    from profile_metrics import FIELDS, save as parse_profile
    rows = []
    for name, mode, period, vectorize in variants():
        path = args.output / name / "gemm_scale" / f"{mode}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        profile = r.get("profile", {})
        metrics = (parse_profile(profile["directory"]) if profile.get("directory") else {})
        rows.append({"variant": name, "status": r.get("status", "unknown"),
                     "correctness": r.get("correctness_passed", False),
                     "period_rows": period, "vectorize": vectorize,
                     "kernel_time_us": r.get("latency_us", "unknown"),
                     "model_latency_cycles": r.get("estimated_latency_cycles", "unknown"),
                     "model_instructions": r.get("instructions", "unknown"),
                     "model_requests": r.get("requests", "unknown"),
                     "model_entries": r.get("entries", "unknown"),
                     "ub_bytes": r.get("final_ir_ub_allocation_bytes", "unknown"),
                     "profile_status": metrics.get("status", "not_collected"),
                     **{key: metrics.get(key, "unknown") for key in FIELDS},
                     "profile_directory": profile.get("directory", "not_collected"),
                     "dma_descriptors": json.dumps(r.get("dma_audit", {}).get("descriptors", [])),
                     "report": str(path)})
    if not rows:
        return
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = [k for k in rows[0] if k != "dma_descriptors"]
    lines = ["# Real-kernel bottleneck controls", "",
             "Model counts are not ISA counts; scalar controls change backend lowering after model scoring.",
             "Profile utilization is not a decomposition into additive phase times.", "",
             "| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    profile_keys = ["variant", "kernel_time_us", "profile_status", *FIELDS]
    with (args.output / "profile_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=profile_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Profile comparison", "",
             "Ratios are fractions (0.5 = 50%); times are microseconds. No phase time is inferred from utilization.",
             "Kernel duration is median per matching record, not summed across repeated launches or cores.", "",
             "| " + " | ".join(profile_keys) + " |", "| " + " | ".join("---" for _ in profile_keys) + " |"]
    lines += ["| " + " | ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                                  for k in profile_keys) + " |" for r in rows]
    (args.output / "profile_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ub-bottleneck-study"))
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.blocks, args.rounds, args.profile_repeats) <= 0:
        parser.error("blocks, rounds and profile repeats must be positive")
    if args.dry_run and args.summarize_only:
        parser.error("--dry-run and --summarize-only cannot be combined")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"blocks": args.blocks, "rounds": args.rounds,
                "profile_repeats": args.profile_repeats, "no_profile": args.no_profile,
                "variants": variants()}
    manifest = json.loads(json.dumps(manifest))
    path = args.output / "plan.json"
    if args.summarize_only:
        if not path.exists() or json.loads(path.read_text()) != manifest:
            parser.error("Use the original experiment arguments to summarize")
        summarize(args)
        return
    path.write_text(json.dumps(manifest, indent=2))
    failures = 0
    for variant in variants():
        cmd = command(args, variant)
        print(cmd, flush=True)
        if args.dry_run:
            continue
        name, mode, _, _ = variant
        (args.output / name / "gemm_scale" / f"{mode}.json").unlink(missing_ok=True)
        with (args.output / f"{name}.log").open("w") as log:
            failures += int(subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode != 0)
        summarize(args)
    if failures:
        raise SystemExit(f"{failures} variants failed; inspect logs under {args.output}")


if __name__ == "__main__":
    main()
