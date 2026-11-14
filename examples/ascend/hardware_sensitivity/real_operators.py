"""Sweep C-1, C, C+1, 2C and original defaults on actual Ascend examples."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys

from benchmark import measure


def candidates(cores, original):
    return sorted({max(1, cores - 1), cores, cores + 1, 2 * cores, original})


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "results.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "passed"]
    groups = sorted({r["group"] for r in rows})
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, max(4, len(groups) * 3.5)), squeeze=False)
    for group, ax in zip(groups, axes[:, 0]):
        data = sorted((r for r in rows if r["group"] == group), key=lambda r: int(r["blocks"]))
        ax.errorbar([int(r["blocks"]) for r in data], [float(r["median_us"]) for r in data],
                    yerr=[[float(r["median_us"]) - float(r["min_us"]) for r in data],
                          [float(r["max_us"]) - float(r["median_us"]) for r in data]], marker="o")
        ax.axvline(int(data[0]["cores"]), color="gray", linestyle="--", label="C")
        ax.set(title=group, xlabel="Logical blocks", ylabel="Median latency (us)")
        ax.set_xticks([int(r["blocks"]) for r in data])
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "real_operators.png", dpi=160)
    fig.savefig(output / "real_operators.svg")
    plt.close(fig)


def run_group(op, size, args, torch, tilelang, aic, aiv, writer, handle):
    from example_gemm import gemm
    from example_vecadd import vector_add

    cores, original = (aiv, 64) if op == "vecadd" else (aic, 32)
    group = f"{op}_{size}"
    if op == "vecadd":
        a = torch.randn(size, device="npu")
        b = torch.randn_like(a)
        expected = a + b
        out = torch.full_like(a, float("nan"))
        logical_bytes, flops = 3 * size * 4, size
    else:
        a = (torch.randn((size, size), device="npu") * 0.125).to(torch.bfloat16)
        b = (torch.randn((size, size), device="npu") * 0.125).to(torch.bfloat16)
        out_dtype = torch.bfloat16 if op == "gemm_cube" else torch.float32
        expected = (a.float() @ b.float().T).to(out_dtype)
        out = torch.full((size, size), float("nan"), device="npu", dtype=out_dtype)
        logical_bytes, flops = 0, 2 * size**3
    block_counts = candidates(cores, original)
    random.Random(args.seed + size).shuffle(block_counts)
    rows = []
    # The same input values, shape, output allocation, tiles and pipeline
    # settings are used across all block candidates in this group.
    for blocks in block_counts:
        row = dict(group=group, operation=op, size=size, cores=cores, original_blocks=original,
                   blocks=blocks, status="failed")
        try:
            if op == "vecadd":
                func = vector_add(size, mode="simt", num_blocks=blocks)
            else:
                func = gemm(size, size, size, dtype="bfloat16",
                            out_dtype="bfloat16" if op == "gemm_cube" else "float32",
                            MIXED=op == "gemm_mixed", num_blocks=blocks)
            kernel = tilelang.compile(func, target={"kind": "ascend", "arch": args.arch}, execution_backend="tvm_ffi")
            (args.output / "sources" / f"{group}_blocks{blocks}.asc").write_text(kernel.get_kernel_source())
            fn = lambda: kernel(a, b, out)
            # Sentinel reset and validation are outside the timed interval.
            for _ in range(2):
                out.fill_(float("nan"))
                fn()
                torch.npu.synchronize()
                tolerance = 0 if op == "vecadd" else (1e-2 if op == "gemm_cube" else 1e-3)
                torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance)
            samples = measure(fn, torch, args)
            median = statistics.median(samples)
            row.update(status="passed", median_us=median, min_us=min(samples), max_us=max(samples),
                       sample_us=json.dumps(samples), logical_bytes=logical_bytes, flops=flops,
                       logical_gbps=logical_bytes / (median * 1000) if logical_bytes else "",
                       tflops=flops / (median * 1e6))
        except Exception as error:
            writer.writerow({**row, "error": repr(error)})
            handle.flush()
            raise
        writer.writerow(row)
        handle.flush()
        rows.append(row)
        print(f"{group}: C={cores}, blocks={blocks}, {median:.3f} us", flush=True)
    at_c = next(r["median_us"] for r in rows if r["blocks"] == cores)
    at_default = next(r["median_us"] for r in rows if r["blocks"] == original)
    for row in rows:
        row["latency_ratio_to_C"] = row["median_us"] / at_c
        row["latency_ratio_to_original"] = row["median_us"] / at_default
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="+", choices=["vecadd", "gemm_cube", "gemm_mixed"],
                        default=["vecadd", "gemm_cube", "gemm_mixed"])
    parser.add_argument("--vector-elements", nargs="+", type=int, default=[2**24])
    parser.add_argument("--gemm-sizes", nargs="+", type=int, default=[2048, 4096])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("real-operators-9579"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
        return
    if any(n <= 0 or n % 8192 for n in args.vector_elements):
        parser.error("Vector element counts must be positive multiples of 8192")
    if any(n < 256 or n % 256 for n in args.gemm_sizes):
        parser.error("GEMM sizes must be positive multiples of 256")
    if min(args.warmup, args.samples, args.repeats) <= 0:
        parser.error("Timing iteration counts must be positive")
    import torch
    import torch_npu  # noqa: F401
    import tilelang

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    props = torch.npu.get_device_properties(args.device)
    aic, aiv = int(props.cube_core_num), int(props.vector_core_num)
    if min(aic, aiv) <= 0:
        raise RuntimeError("Invalid device core counts")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sources").mkdir(exist_ok=True)
    metadata = dict(device=str(props), cube_cores=aic, vector_cores=aiv, tilelang_path=tilelang.__file__,
                    timing="Warm NPU events; no explicit cache flush; per-launch sample-batch averages",
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    for name in ("torch", "torch-npu", "tilelang"):
        try:
            metadata[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metadata[name] = None
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    fields = ["group", "operation", "size", "cores", "original_blocks", "blocks", "status", "median_us",
              "min_us", "max_us", "sample_us", "logical_bytes", "flops", "logical_gbps", "tflops", "error"]
    all_rows = []
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for op in dict.fromkeys(args.operators):
            sizes = args.vector_elements if op == "vecadd" else args.gemm_sizes
            for size in dict.fromkeys(sizes):
                all_rows.extend(run_group(op, size, args, torch, tilelang, aic, aiv, writer, handle))
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "latency_ratio_to_C", "latency_ratio_to_original"])
        writer.writeheader()
        writer.writerows(all_rows)
    if not args.no_plots:
        plot(args.output)


if __name__ == "__main__":
    main()
