"""Direct GM/UB/SIMT pipeline: no layout replay or scheduling bypass."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.instrumentation.session import (
    PassInstrumentationTool,
    register_pass_instrumentation_tool,
    unregister_pass_instrumentation_tool,
)
from tilelang.layout import Fragment, Layout
from tilelang.profiler import do_bench
from tvm import tirx
from layout_diagnostics import plain, summary_fields


def make_kernel(mode, row_stride=1, blocks=32):
    rows, cols, threads = 32 * row_stride, 32, 32
    flat = mode == "ownership"
    shape = (rows * cols,) if flat else (rows, cols)
    pinned = Fragment((32,), forward_thread_fn=lambda i: i, forward_index_fn=lambda i: 0)
    loop_layout = pinned if mode in ("baseline", "padding") else None
    physical = Layout((rows, cols), lambda i, j: i * cols + j + (i // (8 * row_stride)) * 8)

    @T.prim_func
    def main(A: T.Tensor((blocks * rows, cols), "float32"), C: T.Tensor((blocks * 32,), "float32")):
        with T.Kernel(blocks) as block:
            work = T.alloc_shared(shape, "float32")
            result = T.alloc_shared((32,), "float32")
            if mode == "padding":
                T.annotate_layout({work: physical})
            T.copy(A[block * rows : (block + 1) * rows, :], work)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(32, loop_layout=loop_layout):
                    if flat:
                        result[i] = work[i * row_stride * cols] + T.float32(1)
                    else:
                        result[i] = work[i * row_stride, 0] + T.float32(1)
            T.copy(result, C[block * 32 : (block + 1) * 32])

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "baseline", "ownership", "padding", "auto"], default="all")
    parser.add_argument("--row-stride", type=int, choices=[1, 16], default=1)
    parser.add_argument("--case", choices=["single", "reuse", "rotate"], default="single")
    parser.add_argument("--columns", type=int, choices=[16, 32], default=32)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--require-auto-padding", action="store_true", help="fail if auto retains the original storage")
    parser.add_argument("--output", type=Path, default=Path("ub-dma-results"))
    args = parser.parse_args()
    if args.blocks <= 0 or args.iterations <= 0:
        parser.error("blocks and iterations must be positive")
    if args.case == "single" and args.columns != 32:
        parser.error("single uses 32 columns; choose reuse or rotate for 16 columns")
    tilelang.disable_cache()
    rows = 32 * args.row_stride
    torch.manual_seed(0)
    host = torch.randn((args.blocks, rows, args.columns), dtype=torch.float32)
    a = host.reshape(args.blocks * rows, args.columns).to("npu")
    expected = host[:, :: args.row_stride, 0].clone()
    if args.case == "single":
        expected = expected + 1
    else:
        for _ in range(args.iterations):
            expected = expected * 0.999 + 0.001
            if args.case == "rotate":
                expected = torch.roll(expected, shifts=-1, dims=1)
    reference = expected.reshape(-1)
    if args.case != "single":
        from reuse_kernels import make_reuse_kernel
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ("baseline", "ownership", "padding", "auto") if args.mode == "all" else (args.mode,)
    for mode in modes:
        snapshots = []
        seen_passes = []

        @tvm.ir.instrument.pass_instrument
        class Capture:
            def run_after_pass(self, mod, info):
                seen_passes.append(info.name)
                if info.name == "tl.AscendLayoutInference":
                    snapshots.append(mod)

        class CaptureTool(PassInstrumentationTool):
            def create_pass_instrument(self):
                return Capture()

        print(f"{mode}: compiling direct kernel", flush=True)
        # JITKernel builds its own PassContext. Register a compile-scoped tool
        # so callbacks are installed in that context, not just an outer one.
        tool_name = "examples.ub_layout_ablation.dma_layout_capture"
        register_pass_instrumentation_tool(tool_name, CaptureTool)
        try:
            func = (make_kernel(mode, args.row_stride, args.blocks) if args.case == "single" else
                    make_reuse_kernel(mode, args.row_stride, args.blocks, args.columns, args.iterations, args.case == "rotate"))
            kernel = tilelang.compile(func, target="ascend", out_idx=-1,
                                      pass_configs={"tl.ascend_layout_diagnostics": True})
        finally:
            unregister_pass_instrumentation_tool(tool_name)
        if len(snapshots) != 1:
            layout_passes = [name for name in seen_passes if "layout" in name.lower()]
            raise RuntimeError(
                f"Expected one layout inference snapshot, got {len(snapshots)}; "
                f"observed {len(seen_passes)} pass callbacks; layout passes: {layout_passes}"
            )
        padded = []
        layouts_seen = []
        work_seen = []

        def inspect(node):
            if isinstance(node, tirx.SBlock) and "layout_map" in node.annotations:
                for buffer, layout in node.annotations["layout_map"].items():
                    entry = {
                        "buffer": str(buffer.name),
                        "scope": buffer.scope(),
                        "logical_shape": [str(x) for x in buffer.shape],
                        "physical_shape": [str(x) for x in layout.get_output_shape()],
                        "layout": str(layout),
                    }
                    if entry not in layouts_seen:
                        layouts_seen.append(entry)
                    name = str(buffer.name)
                    if buffer.scope() in ("shared", "shared.dyn") and (name == "work" or name.startswith("work_")):
                        if name not in work_seen:
                            work_seen.append(name)
                        # A layout-map entry by itself need not be padding.
                        logical = math.prod(int(x) for x in buffer.shape)
                        physical = math.prod(int(x) for x in layout.get_output_shape())
                        if physical > logical and not any(layout.same_as(x) for x in padded):
                            padded.append(layout)

        for func in snapshots[0].functions.values():
            if isinstance(func, tirx.PrimFunc):
                tirx.stmt_functor.post_order_visit(func.body, inspect)
        # Save the evidence before checking the decision, including on failure.
        (args.output / f"{mode}.tir").write_text(str(snapshots[0]))
        (args.output / f"{mode}.cpp").write_text(kernel.get_kernel_source())
        diagnostics = []
        for func in snapshots[0].functions.values():
            if isinstance(func, tirx.PrimFunc) and func.attrs is not None and "tl.ascend_layout_diagnostics" in func.attrs:
                diagnostics.extend(plain(func.attrs["tl.ascend_layout_diagnostics"]))
        if not diagnostics:
            raise RuntimeError("Layout diagnostics missing; rebuild the TileLang library")
        report = {"case": args.case, "columns": args.columns, "row_stride": args.row_stride,
                  "iterations": args.iterations if args.case != "single" else 1, "blocks": args.blocks, "mode": mode, "padding_applied": bool(padded), "work_layout_entries": work_seen, "layouts": layouts_seen}
        report["layout_diagnostics"] = diagnostics
        report.update(summary_fields(diagnostics))
        (args.output / f"{mode}.json").write_text(json.dumps(report, indent=2))
        if mode == "padding" and not padded:
            raise RuntimeError(f"Explicit padding not observed; inspect {args.output / f'{mode}.json'} and {mode}.tir")
        if mode in ("baseline", "ownership") and padded:
            raise RuntimeError(f"{mode}: unexpected padding; inspect {args.output / f'{mode}.json'}")
        if mode == "auto" and not padded:
            message = f"auto: padding NOT APPLIED (not a padding-benefit validation); inspect {args.output / 'auto.tir'} and auto.json"
            if args.require_auto_padding:
                raise RuntimeError(message)
            print(message, flush=True)
        actual = kernel(a)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-5)
        report["correctness_passed"] = True
        (args.output / f"{mode}.json").write_text(json.dumps(report, indent=2))
        print(f"{mode}: correctness passed; padding={bool(padded)}")
        if args.bench:
            latency = do_bench(lambda: kernel(a), backend="msprof", _n_warmup=10, _n_repeat=30)
            report["latency_us"] = latency * 1000
            (args.output / f"{mode}.json").write_text(json.dumps(report, indent=2))
            print(f"{mode}: {latency * 1000:.3f} us (whole kernel, DMA included)")


if __name__ == "__main__":
    main()
