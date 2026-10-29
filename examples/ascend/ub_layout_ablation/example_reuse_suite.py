"""Sweep direct DMA kernels; each configuration runs in a fresh process."""

import argparse
from pathlib import Path
import subprocess
import sys


CASES = {
    "row128": ("reuse", 32, 1),
    "row64": ("reuse", 16, 1),
    "stride16": ("reuse", 32, 16),
    "rotate": ("rotate", 32, 1),
}


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
    for name, (case, columns, stride) in cases.items():
        for iterations in args.iterations:
            print(f"\n{name}: iterations={iterations}", flush=True)
            cmd = [sys.executable, str(script), "--case", case, "--columns", str(columns),
                   "--row-stride", str(stride), "--iterations", str(iterations),
                   "--blocks", str(args.blocks), "--mode", args.mode,
                   "--output", str(args.output / name / str(iterations))]
            if args.bench:
                cmd.append("--bench")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
