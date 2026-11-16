"""Compare original blocks with core-aware candidates on the current board.

Runs the existing correctness/eager/profiler driver unchanged for each candidate.
This is an experiment, not a change to default operator scheduling.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

from block_config import BLOCK_CASES, POLICIES


def summarize(rows):
    """Keep failures; derive self speedup only against a valid original row."""
    originals = {(r["case"], r["profile"]): r for r in rows if r["block_policy"] == "original"}
    for row in rows:
        row.pop("speedup_vs_original", None)
        original = originals.get((row["case"], row["profile"]))
        if row.get("status") != "passed" or not original or original.get("status") != "passed":
            row["comparison_status"] = "candidate_or_original_unavailable"
            continue
        try:
            baseline = float(original["tilelang_latency_us"])
            candidate = float(row["tilelang_latency_us"])
            if not all(math.isfinite(value) and value > 0 for value in (baseline, candidate)):
                raise ValueError("Invalid latency")
            # All runs must really use the same device and precision/workload.
            for key in ("device_name", "shape", "dtype", "timing"):
                if row.get(key) != original.get(key):
                    raise ValueError(f"Mismatched {key}")
            row["speedup_vs_original"] = baseline / candidate
            row["comparison_status"] = "valid"
        except (KeyError, TypeError, ValueError) as error:
            row["comparison_status"] = f"invalid: {error}"
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=BLOCK_CASES, default=list(BLOCK_CASES))
    parser.add_argument("--profiles", nargs="+", choices=["smoke", "full"], default=["smoke", "full"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-vector-cores", type=int, help="Override target AIV count; actual device unchanged")
    parser.add_argument("--target-cube-cores", type=int, help="Override target Cube count; actual device unchanged")
    parser.add_argument("--output", type=Path, default=Path("blocks-9579"))
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.samples) <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Rounds, warmup, samples and timeout must be positive")
    if any(value is not None and value <= 0 for value in (args.target_vector_cores, args.target_cube_cores)):
        parser.error("Target core counts must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(case, profile, policy) for case in dict.fromkeys(args.cases)
            for profile in dict.fromkeys(args.profiles) for policy in POLICIES]
    random.Random(args.seed).shuffle(jobs)
    manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    job_order=jobs, timing="profiler", repetitions_per_window=1,
                    note="Target-core overrides change generated configurations, not physical resources")
    (args.output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    driver = Path(__file__).with_name("compare.py")
    for index, (case, profile, policy) in enumerate(jobs, 1):
        directory = args.output / "runs" / f"{case}-{profile}-{policy}-{time.time_ns()}"
        directory.mkdir(parents=True)
        command = [sys.executable, str(driver), "--cases", case, "--profile", profile,
                   "--block-policy", policy, "--rounds", str(args.rounds), "--timing", "profiler",
                   "--warmup", str(args.warmup), "--samples", str(args.samples), "--repeats", "1",
                   "--timeout", str(args.timeout), "--device", str(args.device), "--arch", args.arch,
                   "--seed", str(args.seed), "--output", str(directory)]
        target = args.target_vector_cores if case.startswith("vecadd") else args.target_cube_cores
        if target is not None:
            command.extend(["--target-cores", str(target)])
        print(f"[{index}/{len(jobs)}] {case}, {profile}, {policy}: {directory}", flush=True)
        with (directory / "driver.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_path = directory / ("summary.csv" if args.rounds > 1 else "results.csv")
        record = {}
        if result_path.exists():
            with result_path.open() as handle:
                record = next(csv.DictReader(handle), {})
        if not record or result.returncode != 0:
            record.update(status="failed", driver_exit=result.returncode)
        record.update(case=case, profile=profile, block_policy=policy, result_directory=str(directory))
        rows.append(record)
        write_csv(args.output / "results.csv", rows)
    rows = summarize(rows)
    write_csv(args.output / "summary.csv", rows)
    print(f"Results: {args.output / 'summary.csv'}", flush=True)
    if any(row.get("comparison_status") != "valid" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
