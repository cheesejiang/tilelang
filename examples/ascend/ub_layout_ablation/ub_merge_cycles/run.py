#!/usr/bin/env python3
"""Build native SIMT controls and collect measured clock deltas (no model)."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess

from disassemble import disassemble, sha256

CASES = (
    "same_address", "spread_8B", "contiguous_float", "stride_512B",
    "stride_256B", "pair_same_8B", "pair_spread", "pair_conflict",
    "same_address_u64", "contiguous_u64",
    "pair_virtual_group_128B", "stride_128B", "pair_same_key_512B", "stride_16B",
)


ADDRESS_FORMULAS = (
    "0", "8*t", "4*t", "512*t", "256*t", "4*(t%2)", "8*(t%2)",
    "256*(t%2)", "0", "8*t", "128*(t%2)", "128*t", "512*(t%2)", "16*t",
)


def geometry(case):
    """Analytic predictions of the written rules, NOT observed hardware entries.

    All supplied addresses are aligned and <=4B slices have <=64 fragments,
    so capacity cannot add entries beyond the per-group full-line constraint.
    This helper does not call or validate the C++ cost model.
    """
    byte_width = 8 if case in (8, 9) else 4
    addresses = (
        lambda t: 0, lambda t: 8*t, lambda t: 4*t, lambda t: 512*t,
        lambda t: 256*t, lambda t: 4*(t%2), lambda t: 8*(t%2),
        lambda t: 256*(t%2), lambda t: 0, lambda t: 8*t,
        lambda t: 128*(t%2), lambda t: 128*t, lambda t: 512*(t%2), lambda t: 16*t,
    )
    entries = []
    for offset in range(0, byte_width, 4):
        groups = {}
        for lane in range(32):
            address = addresses[case](lane) + offset
            line = address & ~7
            group = ((address >> 3) & 63) & 15
            groups.setdefault(group, set()).add(line)
        entries.append(max(len(lines) for lines in groups.values()))
    return dict(dtype="uint64_t" if byte_width == 8 else "float",
                bytes_per_lane=byte_width, address_formula=ADDRESS_FORMULAS[case],
                model_requests_per_iteration=byte_width // 4,
                model_entries_per_iteration=sum(entries),
                model_fragments_per_request=64)


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


def write_summary(output, rows):
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ("case", "name", "dtype", "bytes_per_lane", "iterations", "samples",
               "address_formula", "model_requests_per_iteration", "model_entries_per_iteration",
               "model_fragments_per_request", "min_cycles", "median_cycles", "max_cycles",
               "median_cycles_per_iteration", "disassembly")
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    table = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(len(CASES))),
                        choices=range(len(CASES)))
    parser.add_argument("--disassemble", action="store_true",
                        help="Extract and disassemble each measured demo; manual review still required")
    parser.add_argument("--disasm-only", action="store_true",
                        help="Disassemble existing builds without recompiling or measuring")
    parser.add_argument("--msobjdump", default="msobjdump", help="CANN ELF extraction tool path")
    parser.add_argument("--objdump", default="llvm-objdump", help="Matching device disassembler path")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if (args.samples < 1 or args.warmups < 0 or
            any(n < 1 or n > 4096 for n in args.iterations)):
        parser.error("samples >= 1, warmups >= 0, iterations in 1..4096 required")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    rows = []
    disasm_failed = False
    if args.disasm_only:
        statuses = []
        for iterations in args.iterations:
            for case in args.cases:
                build = args.output.resolve() / "build" / f"case{case}_n{iterations}"
                if not (build / "demo").is_file():
                    raise FileNotFoundError(f"No measured binary: {build / 'demo'}")
                status = disassemble(build, args.msobjdump, args.objdump)
                statuses.append(dict(case=case, iterations=iterations, disassembly=status))
                print(f"{build.name}: {status}", flush=True)
        (args.output / "disassembly-summary.json").write_text(json.dumps(statuses, indent=2))
        # Do not rewrite old measured summaries or assume they describe these binaries.
        return int(any(r["disassembly"] != "output_generated_unreviewed" for r in statuses))
    for iterations in args.iterations:
        for case in args.cases:
            name = CASES[case]
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
            # Hash the exact binary used for cycles, regardless of disassembly availability.
            (build / "measurement.json").write_text(json.dumps(dict(
                case=case, iterations=iterations, demo_sha256=sha256(build / "demo"),
                samples=args.samples, warmups=args.warmups), indent=2))
            status = (disassemble(build, args.msobjdump, args.objdump)
                      if args.disassemble else "not_requested")
            disasm_failed |= args.disassemble and status != "output_generated_unreviewed"
            row = dict(case=case, name=name, **geometry(case), iterations=iterations,
                       samples=args.samples, min_cycles=min(measured),
                       median_cycles=median, max_cycles=max(measured),
                       median_cycles_per_iteration=median / iterations, disassembly=status)
            rows.append(row)
            write_summary(args.output, rows)
            print(f"{name:24s} N={iterations:4d} median={median:g} "
                  f"cycles ({median / iterations:.3f}/iteration)", flush=True)
    print("\n" + write_summary(args.output, rows))
    print(f"Raw samples and summary: {args.output.resolve()}")
    if disasm_failed:
        print("Disassembly incomplete; cycles retained. Inspect disasm/manifest.json and logs.")
    return int(disasm_failed)


if __name__ == "__main__":
    raise SystemExit(main())
