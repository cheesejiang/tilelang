# Ascend examples

These examples demonstrate complete kernels on Ascend 950. Use the explicit
`tilelang.ascend.language` dialect and an Ascend-enabled build; see the
[backend guide](../../tilelang/ascend/README.md) for setup.
RoPE computes its reference on the CPU. The other examples below run their reference computations on the NPU.

| Example | Purpose |
| --- | --- |
| [GEMM](example_gemm.py) | Persistent matrix multiplication with swizzling, precision modes and a mixed-core epilogue |
| [Split-K GEMM](example_gemm_splitk.py) | Atomic or deterministic reduction across K partitions |
| [Vector addition](example_vecadd.py) | SIMT and explicit SIMD with pipelined data movement |
| [Fused RoPE](example_fused_rope.py) | Native FP16 rotate-half forward and input gradient for the 56-head workload |
| [RMSNorm](example_rmsnorm.py) | SIMT fragments, reduction and pipelined row processing |
| [FP8 quantization](example_per_token_cast_to_fp8.py) | The same group-wise quantization in SIMT and explicit SIMD |

For advanced GEMM implementations and tuning, see [deepgemm](deepgemm).
Attention kernels are in [flash_attention](flash_attention/README.md).

Run an example directly to check correctness and report device performance.
GEMM and attention report latency and TFLOPS; vector addition, RMSNorm and FP8
quantization report latency and effective bandwidth in GB/s.
Bandwidth counts each logical input and output tensor's bytes once, including
auxiliary outputs such as FP8 scales and RMSNorm RSTD.
GEMM throughput uses `2 * M * N * K` operations, including for Split-K.
RoPE reports input/output bandwidth and selects the fastest measured configuration
for each direction with `--tune`. Use `--no-bench` where supported to run only correctness checks.
RoPE's `copy_bw` is its input/output bandwidth divided by the measured bandwidth
of a same-device `torch.Tensor.copy_` between contiguous tensors of the same shape
and dtype. Both measurements count one input read and one output write.
With `--mte2-baseline`, RoPE also measures an input-only GM-to-UB stream for
each tile size. The stream uses the same vector-core assignment and input DMA
geometry, reuses one UB tile per core, and writes one 32-byte sink per core.
`mte2_read` estimates RoPE's
logical GM-to-UB bandwidth including input, positions and coefficient rows.
`mte2_bw` divides that number by the input-only stream bandwidth.
`--dma-baseline` also times a double-buffered TileLang copy with the same tile
assignment. `dma_bw` is the RoPE input/output bandwidth relative to that copy.
The benchmark uses `(600000, 56, 32)` FP16 data and `(4096, 32)` FP16
coefficient tables. Each 128-lane FP16 vector handles four heads. The
benchmark packs the cos and sin tables together before timing, so the
kernel fetches both coefficient rows with one DMA per token.
RoPE uses an offset pipeline and double-buffered inputs and outputs by default.
Add `--no-prefetch` to compare against the previous schedule at the same tile size.

Examples use the configured default target. Set `TILELANG_DEFAULT_TARGET=ascend`
to select AscendC or `TILELANG_DEFAULT_TARGET=pto` to select PTO.

GEMM runs FP8, BF16, FP32, HF32, BF16 output and both accumulation cases.
It uses `M = K = N = 8192`.
Vector addition and FP8 quantization run both SIMT and SIMD by default;
use `--mode simt` or `--mode simd` to select one. Each selected case is timed by default.

```bash
python examples/ascend/example_gemm.py
python examples/ascend/example_gemm_splitk.py --deterministic
python examples/ascend/example_vecadd.py
python examples/ascend/example_per_token_cast_to_fp8.py
python examples/ascend/example_per_token_cast_to_fp8.py --mode simd
python examples/ascend/example_rmsnorm.py --no-bench
python examples/ascend/example_fused_rope.py --tune
python examples/ascend/example_fused_rope.py --tile-tokens 7 --mte2-baseline
python examples/ascend/example_fused_rope.py --tile-tokens 7 --dma-baseline
python -m pytest examples/ascend/test_rmsnorm.py -q
```

Each top-level `example_<name>.py` has one `test_<name>.py`. Correctness tests use
smaller inputs than performance runs. Feature-level and compiler regressions
live in [testing/ascend](../../testing/ascend/README.md), independently of examples.
For copy padding, buffer annotations and scheduling APIs, consult the
[Ascend programming skill](../../.agents/skills/tilelang-ascend/SKILL.md).

The [performance regression driver](../../maint/scripts/ascend_perf_regression.py)
benchmarks these applications, including both vector programming modes, DeepGEMM and attention.
Language-feature tests are not performance entries.
