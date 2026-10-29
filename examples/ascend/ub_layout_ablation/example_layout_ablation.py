"""Four DSL-only UB layout experiments; no compiler switches or pass edits."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics


MODES = ("baseline", "ownership", "padding", "auto")


@dataclass(frozen=True)
class Case:
    name: str
    elements: int
    active: int
    threads: int
    stride: int

    @property
    def period(self):
        return 8 * self.stride

    @property
    def padding_elements(self):
        return 8 if self.stride > 1 else 0


CASES = {
    "contiguous": Case("contiguous", 8192, 8192, 128, 1),
    "row1": Case("row1", 32 * 32, 32, 32, 32),
    "row16": Case("row16", 512 * 32, 32, 32, 16 * 32),
}


def fixed_partition(elements, threads, width):
    from tilelang.layout import Fragment

    return Fragment(
        (elements,),
        forward_thread_fn=lambda i: (i // width) % threads,
        forward_index_fn=lambda i: (i // (threads * width)) * width + i % width,
    )


def analysis_kernel(case, mode, iterations):
    """Pure UB twin: no DMA or observable GM use can veto padding legality."""
    import tilelang.ascend.language as T
    from tilelang.layout import Layout

    n, active, threads, stride = case.elements, case.active, case.threads, case.stride
    init_layout = fixed_partition(n, threads, 2) if mode in ("baseline", "padding") else None
    compute_layout = fixed_partition(active, threads, min(2, active // threads)) if mode in ("baseline", "padding") else None
    # Rank-one storage prevents padding search in the ownership-only twin,
    # while keeping exactly the same flattened logical addresses.
    rank_two = stride > 1 and mode in ("padding", "auto")
    rows, columns = n // 32, 32
    period, pad = case.period, case.padding_elements
    manual_storage = Layout((rows, columns), lambda r, c: r * columns + c + ((r * columns + c) // period) * pad)

    @T.prim_func
    def flat():
        with T.Kernel(1):
            work = T.alloc_shared((n,), "float32")
            with T.SimtVF(threads=threads):
                for i in T.Parallel(n, loop_layout=init_layout):
                    work[i] = T.float32(i)
                for repeat in T.serial(iterations):
                    for i in T.Parallel(active, loop_layout=compute_layout):
                        work[i * stride] = work[i * stride] * T.float32(1.0001) + T.float32(0.001)
                    for i in T.Parallel(active, loop_layout=compute_layout):
                        work[((i + 1) % active) * stride] = work[((i + 1) % active) * stride] * T.float32(1.0001) + T.float32(0.001)

    @T.prim_func
    def matrix():
        with T.Kernel(1):
            work = T.alloc_shared((rows, columns), "float32")
            if mode == "padding":
                T.annotate_layout({work: manual_storage})
            with T.SimtVF(threads=threads):
                for i in T.Parallel(n, loop_layout=init_layout):
                    work[i // columns, i % columns] = T.float32(i)
                for repeat in T.serial(iterations):
                    for i in T.Parallel(active, loop_layout=compute_layout):
                        work[(i * stride) // columns, (i * stride) % columns] = (
                            work[(i * stride) // columns, (i * stride) % columns] * T.float32(1.0001) + T.float32(0.001)
                        )
                    for i in T.Parallel(active, loop_layout=compute_layout):
                        work[(((i + 1) % active) * stride) // columns, (((i + 1) % active) * stride) % columns] = (
                            work[(((i + 1) % active) * stride) // columns, (((i + 1) % active) * stride) % columns]
                            * T.float32(1.0001) + T.float32(0.001)
                        )

    return matrix if rank_two else flat


@dataclass
class Plan:
    init_layout: object
    compute_layout: object
    shifted_layout: object
    padding: int
    physical_elements: int
    metadata: dict


def analyze(case, mode, iterations, artifact_dir):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend import transform as ascend_transform
    from tvm import tirx

    # Only run the Ascend pipeline prefix needed for layout inference. The
    # analysis twin is not an executable kernel: scheduling, InsertSync and
    # code generation are intentionally reserved for the measurement kernel.
    target = tvm.target.Target("ascend")
    program = analysis_kernel(case, mode, iterations).with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": program})
    with target:
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch(
            lower_grid_binding=True,
            lower_thread_binding=False,
            default_threads=None,
            unsupported_annotations=["cluster_dims"],
            launch_dim_tags=["cthread"],
        )(mod)
        # Match AscendPassPipelineBody up to and including layout inference.
        # These examples contain no reducers, dual copies or SIMD operations,
        # but keep their normal prerequisite legalizers in the prefix.
        prefix = (
            tilelang.transform.AddWrapperForSingleBufStore,
            tilelang.transform.LegalizeNegativeIndex,
            tilelang.transform.InjectAssumes,
            ascend_transform.RewriteDualCopy,
            ascend_transform.LegalizeSimdMerging,
            tilelang.transform.Simplify,
            tilelang.transform.CanonicalizeLegacyReducer,
            tilelang.transform.VerifyReducerEpoch,
            tilelang.transform.VerifyBufferInit,
            ascend_transform.UnrollLoopSkipVF,
            tilelang.transform.Simplify,
            ascend_transform.AscendLayoutInference,
        )
        for factory in prefix:
            mod = factory()(mod)
    loops, storage = [], []

    def inspect(node):
        if isinstance(node, tirx.For) and "parallel_loop_layout" in node.annotations:
            loops.append(node.annotations["parallel_loop_layout"])
        if isinstance(node, tirx.SBlock) and "layout_map" in node.annotations:
            for buffer, layout in node.annotations["layout_map"].items():
                if str(buffer.name) == "work":
                    # Inference attaches the same map to several SBlocks.
                    if not any(layout.same_as(existing) for existing in storage):
                        storage.append(layout)

    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, inspect)
    if len(loops) != 3 or len(storage) > 1:
        raise RuntimeError(f"Unexpected analysis structure: {len(loops)} loops, {len(storage)} storage layouts")
    analyzer = tvm.arith.Analyzer()
    padding = 0
    if storage:
        i = tirx.Var("logical_index", "int32")
        analyzer.bind(i, tvm.ir.Range(0, case.elements))
        coords = [i // 32, i % 32]
        mapped = storage[0].map_forward_index(coords)
        if len(mapped) != 1:
            raise RuntimeError("Expected the periodic padding layout to have one physical axis")
        expected = i + (i // case.period) * case.padding_elements
        if not analyzer.can_prove_equal(mapped[0], expected):
            raise RuntimeError(f"Unsupported selected physical layout: {storage[0]}")
        padding = case.padding_elements
    if mode in ("baseline", "ownership") and padding:
        raise RuntimeError(f"{mode} unexpectedly selected padding")
    if mode in ("baseline", "padding"):
        for actual, extent, width in (
            (loops[0], case.elements, 2),
            (loops[1], case.active, min(2, case.active // case.threads)),
            (loops[2], case.active, min(2, case.active // case.threads)),
        ):
            expected = fixed_partition(extent, case.threads, width)
            i = tirx.Var("check_index", "int32")
            analyzer.bind(i, tvm.ir.Range(0, extent))
            if not analyzer.can_prove_equal(actual.map_forward_thread([i])[0], expected.map_forward_thread([i])[0]):
                raise RuntimeError(f"{mode}: pinned ownership changed")
            actual_slots, expected_slots = actual.map_forward_index([i]), expected.map_forward_index([i])
            if len(actual_slots) != len(expected_slots) or not all(analyzer.can_prove_equal(a, b) for a, b in zip(actual_slots, expected_slots)):
                raise RuntimeError(f"{mode}: pinned local element mapping changed")
    physical = case.elements + ((case.elements - 1) // case.period) * padding
    metadata = {
        "case": case.name,
        "mode": mode,
        "threads": case.threads,
        "logical_iteration_element_stride": case.stride,
        "padding_period_elements": case.period,
        "padding_bytes_per_period": padding * 4,
        "logical_work_bytes": case.elements * 4,
        "physical_work_bytes": physical * 4,
        "measurement_ub_bytes_before_alignment": (case.elements + physical + case.active) * 4,
        "init_layout": str(loops[0]),
        "compute_layout": str(loops[1]),
        "shifted_layout": str(loops[2]),
        "storage_layout": str(storage[0]) if storage else "identity",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "analysis.tir").write_text(str(mod))
    (artifact_dir / "plan.json").write_text(json.dumps(metadata, indent=2))
    return Plan(loops[0], loops[1], loops[2], padding, physical, metadata)


def measurement_kernel(case, plan, iterations, blocks):
    """Replay exactly the selected maps; stage DMA through dense buffers."""
    import tilelang.ascend.language as T

    n, active, stride, threads = case.elements, case.active, case.stride, case.threads
    period, pad, physical = case.period, plan.padding, plan.physical_elements
    init_layout, compute_layout, shifted_layout = plan.init_layout, plan.compute_layout, plan.shifted_layout
    output_layout = fixed_partition(active, threads, 1)

    @T.prim_func
    def main(A: T.Tensor((blocks * n,), "float32"), Y: T.Tensor((blocks * active,), "float32")):
        with T.Kernel(blocks) as block:
            incoming = T.alloc_shared((n,), "float32")
            work = T.alloc_shared((physical,), "float32")
            outgoing = T.alloc_shared((active,), "float32")
            T.copy(A[block * n : (block + 1) * n], incoming)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(n, loop_layout=init_layout):
                    work[i + (i // period) * pad] = incoming[i]
                for repeat in T.serial(iterations):
                    for i in T.Parallel(active, loop_layout=compute_layout):
                        work[i * stride + ((i * stride) // period) * pad] = (
                            work[i * stride + ((i * stride) // period) * pad] * T.float32(1.0001) + T.float32(0.001)
                        )
                    for i in T.Parallel(active, loop_layout=shifted_layout):
                        work[((i + 1) % active) * stride + ((((i + 1) % active) * stride) // period) * pad] = (
                            work[((i + 1) % active) * stride + ((((i + 1) % active) * stride) // period) * pad]
                            * T.float32(1.0001) + T.float32(0.001)
                        )
                for i in T.Parallel(active, loop_layout=output_layout):
                    outgoing[i] = work[i * stride + ((i * stride) // period) * pad]
            T.copy(outgoing, Y[block * active : (block + 1) * active])

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--mode", choices=["all", *MODES], default="all")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ub-layout-results"))
    args = parser.parse_args()
    if min(args.iterations, args.blocks, args.samples) <= 0:
        parser.error("iterations, blocks and samples must be positive")
    import tilelang
    from importlib.metadata import PackageNotFoundError, version
    import os
    import platform

    # Always capture fresh compiler decisions, including after C++ rebuilds.
    tilelang.disable_cache()
    cases = list(CASES.values()) if args.case == "all" else [CASES[args.case]]
    modes = MODES if args.mode == "all" else (args.mode,)
    rows = []
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cann_home": os.environ.get("ASCEND_HOME_PATH", os.environ.get("ASCEND_TOOLKIT_HOME")),
        "arguments": {**vars(args), "output": str(args.output)},
    }
    for package in ("torch", "torch-npu", "tilelang"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    if not args.analyze_only:
        import torch
        import torch_npu  # noqa: F401
        from tilelang.profiler import do_bench

        torch.npu.set_device(args.device)
        if not torch.npu.is_available():
            raise RuntimeError("NPU is not available; use --analyze-only for compiler inspection")
        torch.manual_seed(args.seed)
        environment["device"] = torch.npu.get_device_name(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    for case in cases:
        if not args.analyze_only:
            inputs = torch.randn(args.blocks * case.elements, device=f"npu:{args.device}", dtype=torch.float32)
            reference = inputs.reshape(args.blocks, case.elements)[:, :: case.stride][:, : case.active].clone()
            for _ in range(2 * args.iterations):
                reference.mul_(1.0001).add_(0.001)
            reference = reference.reshape(-1)
        prepared = []
        for mode in modes:
            directory = args.output / case.name / mode
            plan = analyze(case, mode, args.iterations, directory)
            row = {**plan.metadata, "iterations": args.iterations, "blocks": args.blocks, "correct": None}
            if not args.analyze_only:
                kernel = tilelang.compile(measurement_kernel(case, plan, args.iterations, args.blocks), target="ascend")
                (directory / "measurement.cpp").write_text(kernel.get_kernel_source())
                actual = torch.empty_like(reference)
                kernel(inputs, actual)
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
                torch.npu.synchronize()
                row["correct"] = True
                if not args.no_bench:
                    prepared.append((row, kernel, actual))
            rows.append(row)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "results.json").write_text(json.dumps(rows, indent=2))
            latency = f"{row['latency_us']:.3f} us" if "latency_us" in row else "not measured"
            status = "analysis only" if row["correct"] is None else "correctness passed"
            print(f"{case.name:12} {mode:10} pad={plan.padding * 4}B/{case.period} elements | {status} | {latency}")
        # Reuse input/output allocations and rotate mode order across rounds.
        # Compilation, golden generation and correctness checks are untimed.
        for sample in range(args.samples if prepared else 0):
            rotation = sample % len(prepared)
            for row, kernel, output in prepared[rotation:] + prepared[:rotation]:
                latency = do_bench(lambda: kernel(inputs, output), backend="msprof", _n_warmup=10, _n_repeat=30) * 1000
                row.setdefault("samples_us", []).append(latency)
        baseline = next((statistics.median(row["samples_us"]) for row, _, _ in prepared if row["mode"] == "baseline"), None)
        for row, _, _ in prepared:
            samples = row["samples_us"]
            row.update(latency_us=statistics.median(samples), spread_us=max(samples) - min(samples))
            if baseline is not None:
                row["speedup_vs_baseline"] = baseline / row["latency_us"]
            print(f"{case.name:12} {row['mode']:10} median={row['latency_us']:.3f} us | spread={row['spread_us']:.3f} us"
                  + (f" | baseline/mode={row['speedup_vs_baseline']:.3f}x" if baseline is not None else ""))
        (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
