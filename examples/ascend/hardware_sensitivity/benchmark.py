"""A5 block-count, working-set reuse and streaming sensitivity experiments.

Device imports are deferred so --help and --plot-only work without an NPU.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess


MIB = 1024**2


def memory_kernel(elements, blocks, tile, rounds, operation):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        a: T.Tensor((elements,), "float32"),
        b: T.Tensor((elements,), "float32"),
        out: T.Tensor((elements,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((tile,), "float32")
            y = T.alloc_shared((tile,), "float32")
            z = T.alloc_shared((tile,), "float32")
            # Each tile always belongs to the same logical block, including
            # across rounds. No inter-block synchronization or shared writes.
            for repeat in T.serial(rounds):
                for task in T.serial(bx, elements // tile, blocks):
                    begin = task * tile
                    T.copy(a[begin : begin + tile], x)
                    if operation == "vecadd":
                        T.copy(b[begin : begin + tile], y)
                        with T.SimtVF(threads=128):
                            for i in T.Parallel(tile):
                                z[i] = x[i] + y[i]
                        T.copy(z, out[begin : begin + tile])
                    else:
                        T.copy(x, out[begin : begin + tile])

    return main


def gemm_kernel(size, blocks):
    import tilelang.ascend.language as T

    tile = 64
    tiles = size // tile

    @T.prim_func
    def main(
        a: T.Tensor((size, size), "float16"),
        b: T.Tensor((size, size), "float16"),
        out: T.Tensor((size, size), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_l1((tile, tile), "float16")
            y = T.alloc_l1((tile, tile), "float16")
            acc = T.alloc_l0c((tile, tile), "float32")
            for task in T.serial(bx, tiles * tiles, blocks):
                row, col = task // tiles, task % tiles
                for k in T.serial(tiles):
                    T.copy(a[row * tile : (row + 1) * tile, k * tile : (k + 1) * tile], x)
                    T.copy(b[col * tile : (col + 1) * tile, k * tile : (k + 1) * tile], y)
                    T.gemm(x, y, acc, transpose_B=True, clear_accum=k == 0)
                T.copy(acc, out[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile])

    return main


def configurations(args, aic, aiv):
    vector_blocks = args.blocks or sorted({1, 4, 8, 16, 28, aiv, 64, 2 * aiv})
    cube_blocks = args.blocks or sorted({1, 4, 8, 16, aic, 32, 2 * aic})
    rows = []
    if args.suite in ("all", "blocks"):
        for op in ("copy", "vecadd"):
            for blocks in vector_blocks:
                rows.append(dict(suite="blocks", operation=op, blocks=blocks,
                                 requested_mib=args.block_working_set_mib, rounds=1))
        for blocks in cube_blocks:
            rows.append(dict(suite="blocks", operation="gemm", blocks=blocks,
                             requested_mib=0, rounds=1))
    for suite in ("reuse", "stream"):
        if args.suite not in ("all", suite):
            continue
        for mib in args.working_sets_mib:
            for op in ("copy", "vecadd"):
                rows.append(dict(suite=suite, operation=op, blocks=args.memory_blocks or aiv,
                                 requested_mib=mib, rounds=1))
    # Randomize once, reproducibly, to reduce systematic sweep-order bias.
    random.Random(args.seed).shuffle(rows)
    return rows


def measure(fn, torch, args):
    for _ in range(args.warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000 / args.repeats)
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError(f"Invalid NPU event measurements: {samples}")
    return samples


def run_case(case, args, torch, tilelang):
    op = case["operation"]
    if op == "gemm":
        n = args.gemm_size
        a = torch.randn((n, n), device="npu", dtype=torch.float16) * 0.125
        b = torch.randn_like(a) * 0.125
        out = torch.full((n, n), float("nan"), device="npu", dtype=torch.float32)
        reference = a.float() @ b.float().T
        func = gemm_kernel(n, case["blocks"])
        footprint = n * n * 8
        logical_bytes = 0  # No claim that unique tensor bytes equal GEMM traffic.
        flops = 2 * n**3
        elements = n * n
    else:
        buffers = 3 if op == "vecadd" else 2
        elements = max(args.tile, int(case["requested_mib"] * MIB) // (4 * buffers) // args.tile * args.tile)
        footprint = elements * 4 * buffers
        rounds = max(1, math.ceil(args.reuse_traffic_mib * MIB / footprint)) if case["suite"] == "reuse" else 1
        case["rounds"] = rounds
        a = torch.randn(elements, device="npu", dtype=torch.float32)
        # Copy uses a dummy aliased argument, not an additional input allocation.
        b = torch.randn_like(a) if op == "vecadd" else a
        out = torch.full_like(a, float("nan"))
        reference = a + b if op == "vecadd" else a
        func = memory_kernel(elements, case["blocks"], args.tile, rounds, op)
        logical_bytes = footprint * rounds
        flops = elements * rounds if op == "vecadd" else 0

    kernel = tilelang.compile(func, target={"kind": "ascend", "arch": args.arch}, execution_backend="tvm_ffi")
    source_path = args.output / "sources" / f"{case['case_id']}.asc"
    source_path.parent.mkdir(exist_ok=True)
    source_path.write_text(kernel.get_kernel_source())
    fn = lambda: kernel(a, b, out)
    fn()
    torch.npu.synchronize()
    torch.testing.assert_close(out, reference, rtol=1e-3 if op == "gemm" else 0,
                               atol=1e-3 if op == "gemm" else 0)
    # Validation-only buffers must not reduce available memory during timing.
    del reference
    samples = measure(fn, torch, args)
    middle = statistics.median(samples)
    row = {**case, "status": "passed", "elements": elements, "working_set_bytes": footprint,
           "logical_bytes": logical_bytes, "flops": flops, "median_us": middle,
           "min_us": min(samples), "max_us": max(samples),
           "logical_gbps": logical_bytes / (middle * 1000) if logical_bytes else "",
           "tflops": flops / (middle * 1e6) if flops else "",
           "sample_us": json.dumps(samples)}
    return row


def plots(output):
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("CSV is available; install matplotlib, then rerun with --plot-only and the same --output") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "results.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "passed"]
    for suite in ("blocks", "reuse", "stream"):
        selected = [r for r in rows if r["suite"] == suite]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for op in sorted({r["operation"] for r in selected}):
            data = sorted((r for r in selected if r["operation"] == op),
                          key=lambda r: float(r["blocks"] if suite == "blocks" else r["working_set_bytes"]))
            x = [float(r["blocks"]) if suite == "blocks" else float(r["working_set_bytes"]) / MIB for r in data]
            axes[0].errorbar(x, [float(r["median_us"]) for r in data],
                            yerr=[[float(r["median_us"]) - float(r["min_us"]) for r in data],
                                  [float(r["max_us"]) - float(r["median_us"]) for r in data]],
                            marker="o", label=op)
            if op != "gemm":
                axes[1].plot(x, [float(r["logical_gbps"]) for r in data], marker="o", label=op)
        for ax in axes:
            ax.set_xlabel("Logical blocks" if suite == "blocks" else "Input + output working set (MiB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("Median latency (us), range across sample batches")
        axes[1].set_ylabel("Logical effective bandwidth (GB/s)")
        fig.suptitle(f"{suite}: warm launches; no explicit cache flush; not DRAM bandwidth")
        fig.tight_layout()
        fig.savefig(output / f"{suite}.png", dpi=160)
        fig.savefig(output / f"{suite}.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "blocks", "reuse", "stream"], default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--blocks", type=int, nargs="+", help="Override block sweep for both AIC and AIV")
    parser.add_argument("--memory-blocks", type=int, help="Reuse/stream block count; defaults to available AIV count")
    parser.add_argument("--working-sets-mib", type=float, nargs="+", default=[8, 16, 32, 64, 96, 128, 192, 256])
    parser.add_argument("--block-working-set-mib", type=float, default=256)
    parser.add_argument("--reuse-traffic-mib", type=float, default=512)
    parser.add_argument("--tile", type=int, default=4096)
    parser.add_argument("--gemm-size", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("hardware-sensitivity-results"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plots(args.output)
        return
    positive = [*args.working_sets_mib, *(args.blocks or []), args.memory_blocks or 1,
                args.block_working_set_mib, args.reuse_traffic_mib, args.tile,
                args.gemm_size, args.samples, args.repeats, args.warmup]
    if any(not math.isfinite(x) or x <= 0 for x in positive):
        parser.error("Sizes, block counts and iteration counts must be finite and positive")
    if args.tile % 128 or args.gemm_size % 64:
        parser.error("--tile must be divisible by 128; --gemm-size must be divisible by 64")

    import torch
    import torch_npu  # noqa: F401
    import tilelang

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    props = torch.npu.get_device_properties(args.device)
    aic, aiv = int(props.cube_core_num), int(props.vector_core_num)
    if min(aic, aiv) <= 0:
        raise RuntimeError("Invalid device core counts")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"device": str(props), "device_name": props.name, "cube_cores": aic,
                "vector_cores": aiv, "tilelang_path": tilelang.__file__,
                "total_memory_bytes": int(props.total_memory),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "environment": {k: os.environ.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_NPU_ARCH")},
                "timing": "NPU events, warm launches, no explicit cache flush; per-launch batch averages"}
    for name in ("torch", "torch-npu", "tilelang"):
        try:
            metadata[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata[name] = None
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    metadata["checkout_commit"] = commit.stdout.strip()
    for tool, flags in (("bisheng", ["--version"]), ("npu-smi", ["info"])):
        executable = shutil.which(tool)
        if executable:
            try:
                result = subprocess.run([executable, *flags], capture_output=True, text=True, timeout=15)
                metadata[tool] = {"path": executable, "output": result.stdout + result.stderr,
                                  "returncode": result.returncode}
            except subprocess.TimeoutExpired:
                metadata[tool] = {"path": executable, "error": "Timed out"}
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    fields = ["case_id", "suite", "operation", "blocks", "requested_mib", "rounds", "status",
              "elements", "working_set_bytes", "logical_bytes", "flops", "median_us", "min_us",
              "max_us", "logical_gbps", "tflops", "sample_us", "error"]
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(configurations(args, aic, aiv)):
            case["case_id"] = f"case_{index:03d}"
            try:
                row = run_case(case, args, torch, tilelang)
                print(f"{case}: {row['median_us']:.3f} us", flush=True)
            except Exception as error:
                writer.writerow({**case, "status": "failed", "error": repr(error)})
                handle.flush()
                raise  # Do not continue after an asynchronous device failure.
            writer.writerow(row)
            handle.flush()
    if not args.no_plots:
        plots(args.output)


if __name__ == "__main__":
    main()
