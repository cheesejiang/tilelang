#!/usr/bin/env python3
"""Build native SIMT controls and collect measured clock deltas (no model)."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess

CASES = (
    "same_address", "spread_8B", "contiguous_float", "stride_512B",
    "stride_256B", "pair_same_8B", "pair_spread", "pair_conflict",
)


def run(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}")
    return result.stdout


def parse_cycles(output):
    if "[Success] Case accuracy verification passed." not in output:
        raise ValueError(f"Accuracy check did not pass:\n{output}")
    matches = re.findall(r"^Cycles per thread: ([0-9 ]+)$", output, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one cycles record:\n{output}")
    cycles = [int(value) for value in matches[0].split()]
    if len(cycles) != 32 or min(cycles) <= 0:
        raise ValueError(f"Invalid warp cycles: {cycles}")
    return cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if (args.samples < 1 or args.warmups < 0 or
            any(n < 1 or n > 4096 for n in args.iterations)):
        parser.error("samples >= 1, warmups >= 0, iterations in 1..4096 required")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    rows = []
    summary = args.output / "summary.csv"
    for iterations in args.iterations:
        for case, name in enumerate(CASES):
            build = args.output.resolve() / "build" / f"case{case}_n{iterations}"
            log = run(["cmake", "-S", str(source), "-B", str(build),
                       "-DCMAKE_ASC_ARCHITECTURES=dav-3510",
                       f"-DSCENARIO_NUM={case}",
                       f"-DREPEAT_TIMES_CONFIG={iterations}"])
            log += run(["cmake", "--build", str(build), "-j2"])
            (build / "build.log").write_text(log)
            records = []
            for sample in range(args.warmups + args.samples):
                output = run([str(build / "demo")])
                (build / f"sample{sample}.log").write_text(output)
                cycles = parse_cycles(output)
                records.append({"warmup": sample < args.warmups,
                                "cycles_per_lane": cycles})
            (build / "samples.json").write_text(json.dumps(records, indent=2))
            # Lanes belong to one warp; they are not independent samples.
            measured = [max(r["cycles_per_lane"]) for r in records if not r["warmup"]]
            median = statistics.median(measured)
            row = dict(case=case, name=name, iterations=iterations,
                       samples=args.samples, min_cycles=min(measured),
                       median_cycles=median, max_cycles=max(measured),
                       median_cycles_per_iteration=median / iterations)
            rows.append(row)
            with summary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{name:24s} N={iterations:4d} median={median:g} "
                  f"cycles ({median / iterations:.3f}/iteration)", flush=True)
    print(f"Raw samples and summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
