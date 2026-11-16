"""TileLang Ascend examples versus explicit Torch-NPU eager baselines.

Each case runs in a separate process. --help/--list need no accelerator imports.
"""

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


CASES = [
    "vecadd_simt", "vecadd_simd", "rmsnorm_fused",
    "gemm_bf16", "gemm_fp32", "gemm_mixed_fp32",
    "deepgemm_bf16", "deepgemm_bmm", "deepgemm_grouped",
    "deepgemm_fp8_quant_matmul", "mha_bf16", "gqa", "gqa_manual", "gqa_backward",
]


class Unavailable(Exception):
    pass


def require(module, name):
    value = getattr(module, name, None)
    if value is None:
        raise Unavailable(f"Required eager interface missing: {name}")
    return value


def fp8_scales(m, n, k, torch, device):
    """Same non-unit K32 E8M0 values in each interface's physical layout.

    TL NT API: A [M,K/64], B [N,K/64] int16. Both logical
    scale tensors have MN-contiguous storage; NT keeps both inputs as given.
    ACL MX: A [M,K/64,2], B [K/64,N,2] raw uint8 E8M0 encodings.
    """
    if k % 64:
        raise ValueError("This benchmark requires K divisible by 64")
    def operand(rows, phase):
        exponent = (torch.arange(rows, device=device)[:, None]
                    + torch.arange(k // 32, device=device)[None, :] + phase) % 3 + 126
        raw = exponent.to(torch.uint8).contiguous()
        # Pack adjacent K32 exponents into one int16, then prepare the
        # MN-contiguous layout expected by the DeepGEMM public API.
        packed = raw.view(torch.int16).T.contiguous().T
        native = raw.view(rows, k // 64, 2)
        decoded = torch.pow(2.0, exponent.float() - 127)
        return packed, native, decoded
    sa, native_a, decoded_a = operand(m, 0)
    sb, native_b, decoded_b = operand(n, 1)
    return sa, sb, native_a, native_b.transpose(0, 1).contiguous(), decoded_a, decoded_b


def compile_kernel(func, args, tilelang, out_idx=-1, fast_math=False):
    cfg = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True} if fast_math else None
    kernel = tilelang.compile(func, target={"kind": "ascend", "arch": args.arch},
                              execution_backend="tvm_ffi", out_idx=out_idx, pass_configs=cfg)
    directory = args.output / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    number = len(list(directory.glob(f"{args.worker}_{args.profile}_*.asc")))
    source = directory / f"{args.worker}_{args.profile}_{number}.asc"
    source.write_text(kernel.get_kernel_source())
    return kernel


def prepare(case, args, torch, torch_npu, tilelang):
    """Return two callable implementations, output validator and metadata."""
    full = args.profile == "full"
    meta = {"comparison_type": "eager_standard", "configuration": "original_default_blocks"}

    def tensor(shape, dtype=torch.float32):
        return (torch.randn(shape, device="npu") * 0.125).to(dtype)

    def close(actual, expected, rtol=1e-3, atol=1e-3):
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
        assert torch.isfinite(actual).all().item()
        assert torch.isfinite(expected).all().item()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    if case.startswith("vecadd"):
        from example_vecadd import vector_add
        n = 2**24 if full else 2**20
        a, b = tensor((n,)), tensor((n,))
        kernel = compile_kernel(vector_add(n, case.removeprefix("vecadd_")), args, tilelang)
        meta.update(shape=[n], dtype="float32", blocks=64, eager_api="torch.add")
        return lambda: kernel(a, b), lambda: torch.add(a, b), lambda x, y: close(x, y, 0, 0), meta

    if case == "rmsnorm_fused":
        from example_rmsnorm import rms_norm_fwd
        batch, d, eps = (4096 if full else 128), 4096, 1e-6
        x, weight = tensor((batch, d)), tensor((d,))
        kernel = compile_kernel(rms_norm_fwd(batch, d), args, tilelang, out_idx=[1, 3])
        tl = lambda: kernel(x.flatten(), weight, eps)
        op = require(torch_npu, "npu_rms_norm")
        eager = lambda: op(x, weight, eps)
        meta.update(comparison_type="eager_fused", eager_api="torch_npu.npu_rms_norm")
        def validate(actual, expected):
            # Independently check both implementations against the FP32 CPU formula.
            xc, wc = x.cpu(), weight.cpu()
            rstd = torch.rsqrt(xc.square().mean(-1) + eps)
            reference = xc * rstd[:, None] * wc
            for result in (actual, expected):
                assert len(result) == 2
                close(result[0].cpu().reshape(batch, d), reference, 1e-4, 1e-4)
                close(result[1].cpu().reshape(batch), rstd, 1e-5, 1e-5)
        meta.update(shape=[batch, d], dtype="float32", epsilon=eps, blocks=64)
        return tl, eager, validate, meta

    if case.startswith("gemm_"):
        from example_gemm import gemm
        n = 4096 if full else 1024
        dtype = torch.bfloat16 if case == "gemm_bf16" else torch.float32
        a, b = tensor((n, n), dtype), tensor((n, n), dtype)
        mixed = case == "gemm_mixed_fp32"
        name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        kernel = compile_kernel(gemm(n, n, n, dtype=name, out_dtype=name, MIXED=mixed), args, tilelang)
        def validate(x, y):
            reference = (a.cpu().float() @ b.cpu().float().T).to(dtype)
            tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
            close(x.cpu(), reference, tolerance, tolerance)
            close(y.cpu(), reference, tolerance, tolerance)
        meta.update(shape=[n, n, n], dtype=name, output_dtype=name, blocks=32,
                    eager_api="torch.mm", hf32=False)
        return lambda: kernel(a, b), lambda: torch.mm(a, b.T), validate, meta

    if case.startswith("deepgemm"):
        from examples.ascend.deepgemm import api
        n, m, k = (1024, 1024, 1024) if full else (256, 256, 256)
        groups = 8 if full else 2
        fp8 = case == "deepgemm_fp8_quant_matmul"
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        if case == "deepgemm_bmm":
            a, b = tensor((groups, m, k), dtype), tensor((groups, n, k), dtype)
            def tl():
                out = torch.empty((groups, m, n), dtype=torch.bfloat16, device="npu")
                api.bf16_bmm(a, b, out)
                return out
            eager = lambda: torch.bmm(a, b.transpose(-1, -2))
            meta.update(shape=[groups, m, n, k], eager_api="torch.bmm")
        elif case == "deepgemm_grouped":
            op = require(torch_npu, "npu_grouped_matmul")
            a, b = tensor((groups * m, k), dtype), tensor((groups, n, k), dtype)
            ends = torch.arange(1, groups + 1, dtype=torch.int64, device="npu") * m
            tl_ends = ends.to(torch.int32)
            # Native weight layout prepared once outside timing, just as TL's
            # already-prepared N,K input is supplied directly.
            native_b = b.transpose(-1, -2).contiguous()
            def tl():
                out = torch.empty((groups * m, n), dtype=torch.bfloat16, device="npu")
                api.m_grouped_bf16_gemm_nt_contiguous(a, b, out, tl_ends)
                return out
            eager = lambda: op([a], [native_b], group_list=ends, split_item=3, group_type=0,
                                output_dtype=torch.bfloat16)[0]
            meta.update(shape=[groups, m, n, k], eager_api="torch_npu.npu_grouped_matmul",
                        comparison_type="eager_fused", note="Group M aligned to 256; both operand layouts prepared")
        else:
            a, b = tensor((m, k), dtype), tensor((n, k), dtype)
            if fp8:
                op = require(torch_npu, "npu_quant_matmul")
                e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
                if e8m0 is None:
                    e8m0 = require(torch, "float8_e8m0fnu")
                sa, sb, native_sa, native_sb, decoded_a, decoded_b = fp8_scales(m, n, k, torch, "npu")
                native_b = b.T
                def tl():
                    out = torch.empty((m, n), dtype=torch.float32, device="npu")
                    api.fp8_gemm_nt((a, sa), (b, sb), out, recipe=(1, 1, 32))
                    return out
                def eager():
                    return op(a, native_b, native_sb, pertoken_scale=native_sa,
                              pertoken_scale_dtype=e8m0, scale_dtype=e8m0,
                              group_sizes=[1, 1, 32], output_dtype=torch.float32)
                # Independent CPU formula validates both layout translations.
                # Casts/dequantization are validation only, never the timed baseline.
                reference = ((a.cpu().float() * decoded_a.cpu().repeat_interleave(32, dim=1))
                             @ (b.cpu().float() * decoded_b.cpu().repeat_interleave(32, dim=1)).T)
                def validate(actual, expected):
                    close(actual.cpu(), reference, 1e-3, 1e-3)
                    close(expected.cpu(), reference, 1e-3, 1e-3)
                meta.update(shape=[m, n, k], output_dtype="float32", dtype=str(dtype),
                            configuration="example_runtime_core_selection", comparison_type="eager_fused",
                            eager_api="torch_npu.npu_quant_matmul", group_sizes=[1, 1, 32],
                            scale_values=[0.5, 1.0, 2.0],
                            scale_layouts={"tilelang_A": list(sa.shape), "tilelang_B": list(sb.shape),
                                           "eager_A": list(native_sa.shape), "eager_B": list(native_sb.shape)},
                            note="FP8 operands and E8M0 K32 scales; both FP32 outputs; scale packing outside timing")
                return tl, eager, validate, meta
            else:
                def tl():
                    out = torch.empty((m, n), dtype=torch.bfloat16, device="npu")
                    api.bf16_gemm_nt(a, b, out)
                    return out
                eager = lambda: torch.mm(a, b.T)
                meta.update(eager_api="torch.mm")
            meta.update(shape=[m, n, k])
        meta.update(dtype=str(dtype), configuration="example_runtime_core_selection")
        return tl, eager, lambda x, y: close(x, y, 1e-2, 1e-2), meta

    if case in ("mha_bf16", "gqa", "gqa_manual", "gqa_backward"):
        from core import flash_attention_fwd
        d = 128
        if case == "mha_bf16":
            s1, heads, skv = (4096 if full else 512), 1, (4096 if full else 512)
        else:
            s1, heads, skv = (1024 if full else 128), (4 if full else 32), (4096 if full else 1024)
        qlen = s1 * heads
        q, k, v = tensor((qlen, d), torch.bfloat16), tensor((skv, d), torch.bfloat16), tensor((skv, d), torch.bfloat16)
        qs = q.view(s1, heads, d).permute(1, 0, 2).unsqueeze(0).contiguous()
        ks, vs = k[None, None, ...], v[None, None, ...]
        scale = 1 / math.sqrt(d)
        blocks = qlen // 128 if case == "mha_bf16" else 32
        meta.update(shape={"S1": s1, "G": heads, "S2": skv, "D": d}, dtype="bfloat16",
                    output_dtype="bfloat16", blocks=blocks, scale=scale,
                    note="MHA uses existing shared builder with BF16 output to match eager; original wrapper defaults to FP32")
        def flatten(result):
            return result.squeeze(0).permute(1, 0, 2).contiguous().view(qlen, d)
        if case == "gqa_backward":
            from core_bwd import flash_attention_bwd_preprocess, flash_attention_bwd_fused_dq_atomic_staged
            fwd = compile_kernel(flash_attention_fwd(qlen, skv, d, num_blocks=32, return_lse=True),
                                 args, tilelang, out_idx=[], fast_math=True)
            delta_kernel = compile_kernel(flash_attention_bwd_preprocess(qlen, d), args, tilelang, out_idx=[], fast_math=True)
            bwd = compile_kernel(flash_attention_bwd_fused_dq_atomic_staged(qlen, skv, d, softmax_scale=scale),
                                 args, tilelang, out_idx=[], fast_math=True)
            do = tensor((qlen, d), torch.bfloat16)
            out, lse = torch.empty_like(q), torch.empty(qlen, device="npu")
            delta = torch.empty(qlen, device="npu")
            fwd(q, k, v, out, lse)
            def tl():
                dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
                delta_kernel(out, do, delta)
                bwd(q, k, v, do, lse, delta, dq, dk, dv)
                return dq, dk, dv
            qs, ks, vs = [t.detach().requires_grad_() for t in (qs, ks, vs)]
            osdpa = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, scale=scale)
            dos = do.view(s1, heads, d).permute(1, 0, 2).unsqueeze(0).contiguous()
            eager = lambda: torch.autograd.grad(osdpa, (qs, ks, vs), dos, retain_graph=True)
            def validate(actual, expected):
                refs = (flatten(expected[0]), expected[1].reshape(skv, d), expected[2].reshape(skv, d))
                for name, x, y in zip(("dQ", "dK", "dV"), actual, refs):
                    assert x.shape == y.shape and x.dtype == y.dtype
                    assert torch.isfinite(x).all().item() and torch.isfinite(y).all().item()
                    error = x.float() - y.float()
                    max_abs = error.abs().max().item()
                    rel_l2 = (error.norm() / y.float().norm().clamp_min(1e-12)).item()
                    assert max_abs < 0.02 and rel_l2 < (0.011 if name == "dQ" else 0.01), (name, max_abs, rel_l2)
            meta.update(comparison_type="eager_composite", eager_api="NPU SDPA + autograd.grad (backward only)",
                        note="TL includes Delta + dQ zero + fused bwd; forward excluded on both sides",
                        eager_grad_fn=type(osdpa.grad_fn).__name__)
            return tl, eager, validate, meta
        func = flash_attention_fwd(qlen, skv, d, num_blocks=blocks, out_dtype="bfloat16", manual_schedule=case == "gqa_manual")
        kernel = compile_kernel(func, args, tilelang, fast_math=True)
        if args.attention_baseline == "fusion":
            op = require(torch_npu, "npu_fusion_attention")
            eager = lambda: op(qs, ks, vs, heads, "BNSD", scale=scale, keep_prob=1.0)[0]
            meta.update(comparison_type="eager_fused", eager_api="torch_npu.npu_fusion_attention")
        else:
            eager = lambda: torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, scale=scale)
            meta.update(eager_api="NPU scaled_dot_product_attention")
        return lambda: kernel(q, k, v), eager, lambda x, y: close(x, flatten(y), 0.05, 0.01), meta
    raise ValueError(case)


def timed(fn, torch, repeats, timing):
    torch.npu.synchronize()
    if timing == "event":
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
    else:
        start_wall = time.perf_counter()
    for _ in range(repeats):
        result = fn()
        del result
    if timing == "event":
        end.record()
        end.synchronize()
        us = float(start.elapsed_time(end)) * 1000 / repeats
    else:
        torch.npu.synchronize()
        us = (time.perf_counter() - start_wall) * 1e6 / repeats
    if not math.isfinite(us) or us <= 0:
        raise RuntimeError(f"Invalid timing: {us}")
    return us


def check_npu_outputs(result, torch):
    if isinstance(result, torch.Tensor):
        assert result.device.type == "npu", f"Unexpected output device: {result.device}"
    elif isinstance(result, (tuple, list)):
        for item in result:
            check_npu_outputs(item, torch)
    else:
        raise TypeError(f"Unexpected output type: {type(result)}")


def worker(args):
    os.environ["TILELANG_DEFAULT_TARGET"] = "ascend"
    os.environ["TILELANG_EXECUTION_BACKEND"] = "tvm_ffi"
    os.environ["ASCEND_NPU_ARCH"] = args.arch
    examples = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(examples), str(examples / "flash_attention"), str(examples.parents[1])]
    row = {"case": args.worker, "status": "failed", "profile": args.profile, "timing": args.timing}
    try:
        import torch
        import torch_npu
        import tilelang
        torch.npu.set_device(args.device)
        torch.manual_seed(args.seed)
        # Control reference FP32 math instead of silently comparing HF32 with FP32.
        matmul = getattr(torch.npu, "matmul", None)
        hf32_controlled = matmul is not None and hasattr(matmul, "allow_hf32")
        if hf32_controlled:
            matmul.allow_hf32 = False
        elif args.worker.startswith(("gemm_fp32", "gemm_mixed")):
            raise Unavailable("Cannot verify/disable Torch-NPU FP32 matmul HF32 mode")
        props = torch.npu.get_device_properties(args.device)
        row.update(device=str(props), tilelang_path=tilelang.__file__, arch=args.arch,
                   torch_version=torch.__version__, hf32=False if hf32_controlled else "unverified")
        for package in ("torch-npu", "tilelang"):
            try:
                row[package + "_version"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        tl, eager, validate, metadata = prepare(args.worker, args, torch, torch_npu, tilelang)
        row.update(metadata)
        # Probe eager before any timed execution. Exceptions are saved as failures;
        # no implicit substitution with another dtype, interface or device.
        reference = eager()
        actual = tl()
        torch.npu.synchronize()
        check_npu_outputs(reference, torch)
        check_npu_outputs(actual, torch)
        validate(actual, reference)
        del reference, actual
        for _ in range(args.warmup):
            tl()
            eager()
        torch.npu.synchronize()
        samples = {"tilelang": [], "eager": []}
        for round_id in range(args.samples):
            for name, fn in (("tilelang", tl), ("eager", eager)) if round_id % 2 == 0 else (("eager", eager), ("tilelang", tl)):
                samples[name].append(timed(fn, torch, args.repeats, args.timing))
        # Recheck fresh results after repeated execution (incl. atomic paths).
        actual, reference = tl(), eager()
        torch.npu.synchronize()
        validate(actual, reference)
        for name, values in samples.items():
            row[name + "_median_us"] = statistics.median(values)
            row[name + "_min_us"] = min(values)
            row[name + "_max_us"] = max(values)
            row[name + "_samples_us"] = values
        row.update(status="passed", correctness="passed",
                   speedup=row["eager_median_us"] / row["tilelang_median_us"])
    except Unavailable as error:
        row.update(status="baseline_unavailable", error=str(error))
    except Exception as error:
        row.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    (args.output / f"{args.worker}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if row["status"] != "failed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--attention-baseline", choices=["fusion", "sdpa"], default="fusion")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--timing", choices=["event", "wall"], default="event")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, default=Path("eager-baseline-9579"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list:
        print("\n".join(CASES))
        return
    if min(args.warmup, args.samples, args.repeats) <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("warmup/samples/repeats/timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        raise SystemExit(worker(args))
    # Never read an old per-case JSON as a result of a failed new run.
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["python"] = sys.executable
    metadata["cann_home"] = os.environ.get("ASCEND_HOME_PATH")
    root = Path(__file__).resolve().parents[3]
    for tool, command in (("checkout_commit", ["git", "-C", str(root), "rev-parse", "HEAD"]),
                          ("bisheng_version", ["bisheng", "--version"]),
                          ("npu_smi", ["npu-smi", "info"])):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            metadata[tool] = {"returncode": result.returncode, "output": result.stdout + result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            metadata[tool] = {"error": str(error)}
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    fields = ["case", "status", "comparison_type", "eager_api", "profile", "shape", "dtype", "blocks",
              "timing", "tilelang_median_us", "eager_median_us", "speedup", "tilelang_min_us", "tilelang_max_us",
              "eager_min_us", "eager_max_us", "tilelang_samples_us", "eager_samples_us", "note", "error"]
    failed = False
    counts = Counter()
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in dict.fromkeys(args.cases):
            path = args.output / f"{case}.json"
            # Preserve an old artifact, if any, but use a distinct fresh worker
            # directory so it can never be mistaken for the new measurement.
            run_dir = args.output / "runs" / f"{case}_{time.time_ns()}"
            run_dir.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                       "--profile", args.profile, "--device", str(args.device), "--arch", args.arch,
                       "--timing", args.timing, "--warmup", str(args.warmup), "--samples", str(args.samples),
                       "--repeats", str(args.repeats), "--seed", str(args.seed), "--output", str(run_dir),
                       "--attention-baseline", args.attention_baseline]
            print(f"Running {case} ...", flush=True)
            try:
                with (args.output / f"{case}.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                fresh = run_dir / f"{case}.json"
                row = json.loads(fresh.read_text()) if fresh.exists() else dict(case=case, status="worker_crashed", error=f"exit={result.returncode}")
            except subprocess.TimeoutExpired:
                row = dict(case=case, status="timeout", error=f"Exceeded {args.timeout}s")
            path.write_text(json.dumps(row, indent=2))
            encoded = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            writer.writerow(encoded)
            handle.flush()
            failed |= row["status"] not in ("passed", "baseline_unavailable")
            counts[row["status"]] += 1
            print(f"{case}: {row['status']}" + (f", speedup={row['speedup']:.3f}x" if row.get("speedup") else ""), flush=True)
    print(f"Results: {args.output / 'results.csv'}")
    print(f"Summary: {dict(counts)}; baseline_unavailable is NOT a performance pass")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
