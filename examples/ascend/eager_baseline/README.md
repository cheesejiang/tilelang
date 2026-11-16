# TileLang versus Torch-NPU eager on Ascend 9579

Initial runnable comparison suite. This uses the existing Ascend examples,
not the earlier simplified hardware-sensitivity kernels. No backend changes
or automatic block tuning are performed. It is **not a direct ops-nn benchmark**.

## First run

Run from the repository root in the 9579 machine's built TileLang environment.
Copy this directory together with the current examples it imports.

```bash
# Check a small set before compiling all categories.
python examples/ascend/eager_baseline/compare.py \
  --cases vecadd_simt vecadd_simd gemm_bf16 rmsnorm_fused \
  --profile smoke --warmup 5 --samples 3 --repeats 5 \
  --output eager-smoke

# Full case list, still using smaller smoke shapes.
python examples/ascend/eager_baseline/compare.py \
  --profile smoke --output eager-all-smoke

# Larger shapes, after inspecting smoke results.
python examples/ascend/eager_baseline/compare.py \
  --profile full --output eager-full
```

`--list` prints case names without importing Torch or TileLang. Use `--cases`
to run any subset. `--attention-baseline sdpa` explicitly switches forward
attention to NPU SDPA; it never silently falls back from the fusion API.

Each case runs in a separate Python process with a default 900-second timeout.
A failure is logged; the parent preserves other results. A hung/crashed device
can still require external recovery before subsequent processes can run.

## Implemented comparisons

| Cases | Eager baseline | Type / constraints |
|---|---|---|
| `vecadd_simt`, `vecadd_simd` | `torch.add` | standard; FP32, original 64 blocks |
| `rmsnorm_fused` | `torch_npu.npu_rms_norm` | fused; FP32, epsilon=1e-6, validates Y and RSTD; unsupported FP32 is reported, not cast |
| `rmsnorm_composite` | square/mean/add/rsqrt/mul/mul | composite; separate explicit alternative |
| `gemm_bf16` | BF16 `torch.mm` | standard; BF16 output on both sides, original 32 blocks |
| `gemm_fp32`, `gemm_mixed_fp32` | FP32 `torch.mm` | standard; FP32 inputs/outputs, HF32 disabled; mixed explicitly enabled for TL |
| `splitk_atomic`, `splitk_deterministic` | BF16→FP32 casts + FP32 `torch.mm` | composite, FP32 output; casts included; **not native BF16 matmul throughput** |
| `quant_simt`, `quant_simd` | FP32 view/abs/amax/clamp/scale/FP8 cast | composite; 128-element groups, amax floor 1e-4, FP32 scale; includes zero row |
| `deepgemm_bf16` | BF16 `torch.mm` | standard; BF16 output on both sides |
| `deepgemm_bmm` | BF16 `torch.bmm` | standard; BF16 output, transpose view pre-created by operands |
| `deepgemm_grouped` | `torch_npu.npu_grouped_matmul` | fused candidate; M-group ends and supported output dtype must pass installed API probe; no silent Python-loop fallback |
| `deepgemm_fp8_composite` | FP8→FP32 casts + FP32 `torch.mm` | composite with **unit K32 scales only**, not a quant_matmul or FP8 peak-throughput comparison |
| `mha_bf16` | `npu_fusion_attention`, or explicitly selected SDPA | uses existing attention builder with **BF16 output**, unlike original MHA wrapper's FP32 output; comparison configuration is recorded |
| `gqa`, `gqa_manual` | `npu_fusion_attention`, or explicitly selected SDPA | BF16 output, same logical GQA, 32 TL blocks; pre-created contiguous Q head layout outside timing |
| `gqa_backward` | NPU SDPA `autograd.grad` | composite/framework backward; forward excluded; TL timing includes Delta + dQ zero + fused backward; no accumulated `.grad` |

All direct interface support is provisional until run on the installed
Torch-NPU/CANN version. A missing API gets `baseline_unavailable`, no speedup.
Other exceptions, including unsupported dtype/shape, numerical errors and device
errors, are recorded as `failed` with traceback rather than guessed from error
strings or silently substituted. Inspect the log to determine the cause.

The first version does **not** claim an installed, validated direct
`npu_quant_matmul` match for DeepGEMM's UE8M0 representation, nor a direct FP32
`npu_dynamic_block_quant` counterpart. Composite rows are clearly named. It also
does not benchmark every DeepGEMM layout/epilogue/random-rounding variant; those
remain in the full correctness baseline.

## Shapes

| Family | smoke | full |
|---|---|---|
| VecAdd | N=2^20 | N=2^24 |
| RMSNorm | batch=128,d=4096 | batch=4096,d=4096 |
| GEMM | 1024³ | 4096³ |
| Split-K | M=N=512,K=1024,split=4 | M=N=512,K=8192,split=4 |
| Quant | M=128,N=8192 | M=N=8192 |
| DeepGEMM dense | 256³ | 1024³ |
| DeepGEMM grouped/BMM | groups=2,M=N=K=256 | groups=8,M=N=K=1024 |
| MHA | S=512,D=128 | S=4096,D=128 |
| GQA/manual/backward | S1=128,G=32,S2=1024,D=128 | S1=1024,G=4,S2=4096,D=128 |

Attention full shapes are deliberately smaller than the examples' largest
8192/G32 configuration, to establish interface compatibility first. Do not
compare them against historical results with different shapes or output types.

## Measurement contract

- Correctness is checked before and after timed repeated execution. GEMM and
  RMSNorm have CPU FP32 references; quant/attention/DeepGEMM are checked against
  their stated eager computation, using existing example-style tolerances.
- Default `--timing event` uses NPU event intervals around a batch of launches.
  Host enqueue gaps/device waits can contribute; this is not profiler-isolated
  instruction time. `--timing wall` separately measures synchronized host elapsed
  time, including host API/allocator overhead. Never merge the two metrics.
- Both callables allocate their output(s) on each call. Necessary casts,
  reductions, output preparation, and all composite operators are timed.
  Compilation, input generation, reference validation and warmup are excluded.
- Shared input values are prepared once; eager and TL can use separate,
  pre-prepared layouts, recorded where applicable. Layout preparation costs are
  excluded, so this is not a complete application-layout conversion benchmark.
- Warm-cache, no explicit flush. Each sample batch runs `--repeats` calls.
  Sample order alternates TL/eager and eager/TL. Defaults: warmup=10, samples=5,
  repeats=20. Repeat a session into another directory for stability checks.
- Default Torch FP32 matmul HF32 is disabled if the installed control exists.
  FP32-sensitive cases are marked unavailable if that control cannot be verified.
- Check profiling separately if exact lower-level library/operator dispatch is
  required. NPU output checks do not prove that no internal CPU fallback occurred.
- Input generation and tensor allocation can require more memory than the
  nominal input/output size; full quantization and backward are not memory-limit
  tests. No GM auto-splitting is implemented.

## Files to return

- `results.csv`: status, baseline type/API, shape, times, samples, speedup.
- `environment.json`: command, Python, CANN home, available tool versions, commit.
- `<case>.json`: device/package/import details and complete per-case metadata.
- `<case>.log`: compiler messages and traceback; send this for failed cases.
- `runs/`: fresh worker directories including generated sources where compiled
  through this wrapper. Existing JIT/DeepGEMM internal compilation is not exported.

`speedup = eager_median_us / tilelang_median_us`. Above 1 means TL faster **than
that named baseline**, not necessarily faster than a native fused kernel.
The report distinguishes passed, unavailable, failed, crash and timeout. No
speedup is emitted for an unavailable/failed comparison. A previous result is
never reused as the result of a failed new measurement.

This initial version has host-side syntax/driver checks only until run on the
9579. It adds benchmark tooling; it does not change TileLang backend code.
