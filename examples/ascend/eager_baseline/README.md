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
| `gemm_bf16` | BF16 `torch.mm` | standard; BF16 output on both sides, original 32 blocks |
| `gemm_fp32`, `gemm_mixed_fp32` | FP32 `torch.mm` | standard; FP32 inputs/outputs, HF32 disabled; mixed explicitly enabled for TL |
| `deepgemm_bf16` | BF16 `torch.mm` | standard; BF16 output on both sides |
| `deepgemm_bmm` | BF16 `torch.bmm` | standard; BF16 output, transpose view pre-created by operands |
| `deepgemm_grouped` | `torch_npu.npu_grouped_matmul` | fused candidate; M-group ends and supported output dtype must pass installed API probe; no silent Python-loop fallback |
| `deepgemm_fp8_quant_matmul` | `torch_npu.npu_quant_matmul` | fused MX FP8 E4M3FN, K32 E8M0 scales, FP32 outputs; non-unit scales 0.5/1/2 |
| `mha_bf16` | `npu_fusion_attention`, or explicitly selected SDPA | uses existing attention builder with **BF16 output**, unlike original MHA wrapper's FP32 output; comparison configuration is recorded |
| `gqa`, `gqa_manual` | `npu_fusion_attention`, or explicitly selected SDPA | BF16 output, same logical GQA, 32 TL blocks; pre-created contiguous Q head layout outside timing |
| `gqa_backward` | NPU SDPA `autograd.grad` | composite/framework backward; forward excluded; TL timing includes Delta + dQ zero + fused backward; no accumulated `.grad` |

All direct interface support is provisional until run on the installed
Torch-NPU/CANN version. A missing API gets `baseline_unavailable`, no speedup.
Other exceptions, including unsupported dtype/shape, numerical errors and device
errors, are recorded as `failed` with traceback rather than guessed from error
strings or silently substituted. Inspect the log to determine the cause.

The FP8 case calls `npu_quant_matmul` directly with `group_sizes=[1,1,32]`,
raw uint8 E8M0 scales and explicit `scale_dtype`/`pertoken_scale_dtype`.
It never falls back to FP32 matmul. API absence is recorded as unavailable;
installed-version signature or device errors are recorded with a traceback.
This suite does not benchmark every DeepGEMM layout/epilogue/random-rounding
variant; those remain in the full correctness baseline.

### FP8 direct comparison

```bash
python examples/ascend/eager_baseline/compare.py \
  --cases deepgemm_fp8_quant_matmul --profile smoke \
  --output eager-fp8-smoke

python examples/ascend/eager_baseline/compare.py \
  --cases deepgemm_fp8_quant_matmul --profile full \
  --output eager-fp8-full
```

Both operands use FP8 E4M3FN. Per-K32 scale values vary across rows and K groups
among 0.5, 1, and 2; these are exactly representable in E8M0. Both outputs are
FP32. An independent CPU FP32 dequantization/matmul reference checks each side
before/after timing. CPU reference work is not timed.

Scale layouts are prepared outside timing from the same exponent values:

- TileLang NT API: A `[M,K/64]` packed int16, M-contiguous physical storage;
  B `[N,K/64]` packed int16, N-contiguous physical storage. The public
  NT API keeps these logical shapes; it transposes scales internally for the kernel.
- Native API: A `[M,K/64,2]`, B `[K/64,N,2]`, uint8 E8M0 encodings, interpreted
  using the explicit E8M0 dtype arguments. Weight and B scales both retain
  transposed views: do not make only the B scales contiguous, since the native
  interface requires their transpose layouts to agree.

The former composite case's scale dimensions did not match the public
DeepGEMM API. This replacement corrects the layouts; an unequal-M/N CPU test
checks byte packing, transposition, and decoded values. It does not establish
NPU kernel compatibility. Any operand-format/version restriction on the board
must be inspected from that case's log; no silent format/type substitution is
performed. Source: [official quant_matmul documentation](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md).

## Shapes

| Family | smoke | full |
|---|---|---|
| VecAdd | N=2^20 | N=2^30 |
| RMSNorm | batch=128,d=4096 | batch=4096,d=4096 |
| GEMM | 1024³ | 8192³ |
| DeepGEMM BF16 dense | 256³ | 4096³ |
| DeepGEMM FP8 dense | 256³ | M=1024,N=4096,K=7168 |
| DeepGEMM grouped/BMM | groups=2,M=N=K=256 | groups=8,M=N=K=1024 |
| MHA | S=512,D=128 | S=4096,D=128 |
| GQA/manual/backward | S1=128,G=32,S2=1024,D=128 | S1=8192,G=32,S2=8192,D=128 |

Full shapes follow the original examples' performance entry defaults for
VecAdd, RMSNorm, GEMM, MHA and GQA/manual/backward. DeepGEMM dense follows
`maint/scripts/ascend_perf_regression.py` (BF16 4096³ and FP8 1024x4096x7168).
BMM/grouped have no corresponding fixed original performance entry, so their
representative groups=8, M=N=K=1024 shapes are retained. Only shapes are aligned:
existing comparison dtypes, output semantics and timing contracts are unchanged.
Do not compare historical full results with these new shapes as one workload.

## Measurement contract

- Correctness is checked before and after timed repeated execution. GEMM and
  RMSNorm have CPU FP32 references; quant/attention/DeepGEMM are checked against
  their stated eager computation, using existing example-style tolerances.
- Default `--timing profiler` follows the measurement portion of TileOPs'
  Ascend protocol: 3 warmups per side, 15 samples per side, one complete call
  per profiler window. A 1 GiB cache-eviction kernel and synchronization precede
  every warmup/sample, outside the window. TL/eager order alternates by sample.
- Timing is the union of device kernel execution intervals, covering all kernels
  in the callable and counting overlapping AIC/AIV activity only once. Host
  launch gaps and Python/allocator overhead are not part of this metric.
  Within-call reuse is preserved; cache eviction is an experimental policy,
  not proof of complete eviction of every cache level on every SKU.
- Both callables allocate outputs on each call. Necessary device casts,
  reductions, output preparation, zeroing and composite kernels remain inside
  the profiled callable. Compilation, input generation, validation and initial
  setup are outside measurement.
- Raw profiler traces and parsed per-sample records are retained. Missing raw
  records, unresolvable flush calibration or flush work inside a measured
  window fail the case; no automatic event/CPU timing fallback is used.
  The raw parser follows TileOPs' CANN 9.2 FFTS record format and must be verified
  against the installed profiler version on the board.
- Legacy `--timing event` and `--timing wall` remain explicit alternatives, with
  no cache eviction. Specify their desired `--samples/--repeats/--warmup`;
  do not mix their results with profiler measurements. Profiler mode requires
  `--repeats 1`. The defaults are now warmup=3, samples=15, repeats=1.
- Shared values and API-specific layouts are prepared before measurement. This
  is not a full application-layout conversion benchmark.
- Default Torch FP32 matmul HF32 is disabled if the installed control exists.
  FP32-sensitive cases are marked unavailable if that control cannot be verified.
- Check profiling separately if exact lower-level library/operator dispatch is
  required. NPU output checks do not prove that no internal CPU fallback occurred.
- Input generation and tensor allocation can require more memory than the
  nominal input/output size; full-size workloads are not memory-limit
  tests. No GM auto-splitting is implemented.

## Files to return

- `results.csv`: status, baseline type/API, shape, times, samples, speedup.
- `environment.json`: command, Python, CANN home, available tool versions, commit.
- `<case>.json`: device/package/import details and complete per-case metadata.
- `<case>.log`: compiler messages and traceback; send this for failed cases.
- `runs/`: fresh worker directories including generated sources where compiled
  through this wrapper. Existing JIT/DeepGEMM internal compilation is not exported.

`speedup = eager_latency_us / tilelang_latency_us`. In profiler mode latency
is the arithmetic mean of the 15 device-busy samples; median/min/max/stdev
remain descriptive fields. Above 1 means TL faster **than
that named baseline**, not necessarily faster than a native fused kernel.
The report distinguishes passed, unavailable, failed, crash and timeout. No
speedup is emitted for an unavailable/failed comparison. A previous result is
never reused as the result of a failed new measurement.

The earlier FP8 case was confirmed working on 9579; this new profiler timing
path has host-side checks only until separately verified on the board. It adds benchmark tooling; it does not change TileLang backend code.

Split-K GEMM and FP8 group quantization are excluded from this performance
comparison suite: their cast/composite baselines do not provide the desired
fused-operator comparison. Their original correctness tests remain included.

RMSNorm uses only the validated fused `npu_rms_norm` baseline; the separate
eager-composite RMSNorm case has been removed.

## Independent rounds

The driver defaults to `--rounds 3`. Each round starts a fresh suite process,
and each case runs in its own worker process. Each round repeats correctness,
3 cold-cache warmup calls per side, 15 alternating single-call profiler
samples per side, and post-timing correctness checks. Rounds do not clear
compiler caches or reset the device. Device clocks are not locked by this
driver; run on an idle device with a consistent clock/power policy.
Use `--rounds 1` for a single suite, or override `--warmup/--samples/--repeats`.

```bash
python examples/ascend/eager_baseline/compare.py \
  --profile full --rounds 3 --output eager-full-three-rounds
```

Each `round-XX-<timestamp>/` contains its own environment.json, results.csv,
suite.log, case logs, JSON and generated sources. Top-level results.csv retains
all round rows. summary.csv reports per-case arithmetic mean of the round mean latencies in
profiler mode, `speedup = mean(eager round means) / mean(TileLang round means)`,
and min/max of individual round speedups. Any missing, failed, timed-out or
unavailable round prevents a valid aggregate speedup; it is not dropped.
rounds.json records the protocol arguments and summary. Cross-SKU retention
and pass thresholds are not calculated automatically.

Reference: local TileOPs `benchmarks/ascend/timing.py` and `profile_parser.py`.
The default reproduces their timing/cache/statistics method. TileOPs reservation
and clock-management infrastructure are not imported, and the protocol is named
`ascend-profiler-cold-cache-v1` rather than claiming its locked-clock protocol.
Re-establish the 9579 baseline; earlier event results are not comparable.

`results.csv` includes `device_name` (the actual device properties name),
`device` and `arch`; `summary.csv` includes `device_name` for cross-SKU comparison.
Conflicting device names across rounds invalidate the aggregate.

## 9579 block-configuration experiment

`block_tuning.py` compares the original configurations against pre-lowering
core-aware configurations for VecAdd SIMT/SIMD and ordinary GEMM
BF16/FP32/mixed-FP32. No operator default or framework pass is changed.

Policies: `original` (64 for VecAdd, 32 for GEMM), `auto` (minimum of target
core count and output task count), `c-1`, `c`, `c+1`, `2c`. VecAdd queries AIV
count and has N/8192 tasks; GEMM queries Cube count and has (M/256)*(N/256)
output tiles. Only `auto` clamps to task count; other candidates intentionally
exercise idle/oversubscribed blocks. Each candidate reconstructs the DSL with
its B so indexing, persistent loops and launch count agree.

Start with a small experiment:

```bash
python examples/ascend/eager_baseline/block_tuning.py \
  --profiles smoke --rounds 1 --output blocks-smoke
```

Then run both existing shape profiles, three independent rounds per candidate:

```bash
python examples/ascend/eager_baseline/block_tuning.py \
  --profiles smoke full --rounds 3 --output blocks-9579
```

Each candidate reuses compare.py's correctness/profiler protocol (3 warmups,
15 samples per side, 1 GiB eviction), original dtypes/tiles/pipelines and eager
reference. Jobs run in reproducibly shuffled order. Same case/profile uses
identical seed and values across policies. Each candidate/round/case is in a
fresh process; raw profiles, logs, environment and sources remain under runs/.
The original and `auto` policies are both measured even when their B matches
another candidate; equality does not imply identical noisy timing measurements.

Top-level `summary.csv` records actual physical cores, target cores, policy,
chosen blocks, tasks, shape, latencies and two distinct ratios:

- `speedup`: eager latency / candidate TileLang latency.
- `speedup_vs_original`: original TileLang latency / candidate TileLang latency;
  >1 means improvement over the original configuration. Baseline failure or
  candidate failure prevents this comparison; failures remain in the report.

Evaluate each case/shape separately. Candidate-best results are exploratory,
not proof of a universally optimal policy. Validate a chosen policy with a
separate output run; do not hide regressions or replace original baseline rows.

For a single policy without the sweep:

```bash
python examples/ascend/eager_baseline/compare.py \
  --cases vecadd_simt gemm_bf16 --profile smoke --block-policy auto \
  --rounds 1 --output blocks-auto
```

Explicit target specifications can override device-query input:
`block_tuning.py --target-vector-cores 48 --target-cube-cores 24 ...`, or
`compare.py --cases gemm_bf16 --block-policy auto --target-cores 24 ...`.
Actual device core counts are still recorded separately. This checks generated
configurations on 9579, not performance of another SKU. Configuration selection
is host-only (block_config.py), but these benchmark commands require an NPU
for correctness and timing and do not implement a standalone offline compiler.

## Logical-block to physical-core diagnosis

Run the independent small BF16 Cube GEMM diagnostic, without external msprof:

```bash
python examples/ascend/eager_baseline/diagnose_block_mapping.py \
  --blocks 16 28 29 32 56 --size 1024 --output block-mapping
```

For the first comparison only:

```bash
python examples/ascend/eager_baseline/diagnose_block_mapping.py \
  --blocks 28 29 --output block-mapping-28-29
```

The driver uses one fresh process per blocks value and prints all records,
grouped by raw `asc_get_core_id()` and sorted by entry time. Copy the console
text back for analysis; no binary archive is required. A new output directory
is required for reruns. Each blocks directory contains `run.log`, `records.csv`,
`metadata.json`, `original.asc` and `instrumented.asc`; the root `records.csv`
combines configurations. Stop on errors rather than hide a missing record.

This reuses the original BF16 Cube GEMM DSL, then instruments the generated
source in a process-local Ascend postproc callback. It does not change operator
defaults, compiler passes or the original performance suite. Kernel caching is
disabled so a cached uninstrumented kernel cannot silently run.

The output tensor has a larger backing allocation: its MxN BF16 view is followed
by 128 bytes of record storage per logical block. The callback addresses this
non-overlapping tail through the existing C pointer, preserving the launch ABI.
Each block records logical ID, raw core ID, entry/completion system cycles,
issued output-tile count, and exit core ID. The tile counter increments only
inside the existing guarded output-store path, including persistent iterations.
Empty blocks therefore produce records with zero output tiles. CPU checks
validate all expected records, task counts, timestamp order and output values.

Entry is after `asc_init`. Completion is after the original body and an added
`asc_sync()` to drain device pipelines; record publication occurs afterwards.
An explicit data-cache flush publishes scalar GM records. A 1 GiB eviction
precedes the one diagnostic call. There is no warmup, eager comparison, internal
profiler or speedup claim. Instrumentation and the fence perturb execution;
use this to examine mapping/order, not as replacement baseline timings.

The raw core ID is preserved, with no assumption that it equals logical block
ID or a dense 0..27 index. Ascend950's system counter is 1 GHz (1000 ticks/us),
independent of the 1650 MHz compute frequency. Raw cycles are retained alongside
relative microseconds; `--ticks-per-us` is available for explicit conversion.
Reference: local asc-devkit docs for `asc_get_core_id` and
`asc_get_system_cycle` under `docs/zh/api/SIMD-API/c_api/utils/sys_var/`.

Requires the matching CANN/toolchain already used for these examples on 9579.
Host checks and a local CANN scalar-record API compilation probe do not replace
an on-board test of the full instrumented GEMM.
# Markdown 输出

## 公共 AutoBlocks Pass 的全量性能验证

```bash
python examples/ascend/eager_baseline/compiler_blocks.py \
  --profiles smoke full --rounds 3 --warmup 3 --samples 15 \
  --output compiler-blocks-all
```

默认覆盖本目录全部 14 个用例，对比公共编译 Pass 的关闭/开启状态。
不在 Python 中预先调整 blocks；每侧均先验证正确性，再使用现有 Profiler 协议计时。
输出 `summary.md` 和 CSV，记录实际改写、blocks、耗时、相对关闭状态的收益和 eager S。
未被优化的用例也保留在结果中。多 kernel 用例的 blocks 以列表记录。
该验证模式关闭完整 kernel 编译缓存以取得实际 Pass 记录，编译不计入耗时。
快速检查可改为 `--rounds 1 --samples 3`。

也可以单独执行一侧，例如：

```bash
python examples/ascend/eager_baseline/compare.py \
  --profile full --compiler-auto-blocks on --output compiler-on-full
```

`--compiler-auto-blocks` 不能与实验性 `--block-policy` 或 `--target-cores` 混用。
详细口径见 `docs/compiler_internals/ascend_auto_blocks.md`。

Attention forward 合法 blocks 对比（保持输入、tile 大小、tile 总数及原有 DSL 不变）：

```bash
python examples/ascend/eager_baseline/block_tuning.py \
  --cases mha_bf16 gqa gqa_manual --profiles smoke full \
  --rounds 1 --samples 3 --output attention-blocks16
```

每个算子对比 `original`、`auto` 与合法的 `fixed-16`，结果自动写入 `summary.csv` 和 `summary.md`。
Attention 的 auto 取不超过 `min(query tile 总数, 目标 Cube 核数)` 的最大因数，
保证 query tile 总数能被 blocks 整除。这与 VecAdd/GEMM 的直接取 min 策略不同。
9579 的 28 Cube 核下，smoke MHA 自动选 4 blocks，full MHA、smoke/full GQA/manual 自动选 16。
full MHA 的 32 个 query tile 从每 block 1 个变为 2 个；full GQA/manual 的
2048 个 query tile 从每 block 64 个变为 128 个。三者原始配置均为 32 blocks。
smoke MHA 仅有 4 个 query tile，执行 original 和 auto，
不执行不合法的 fixed-16，Markdown 对该项标记未运行。smoke GQA/manual 的 32 个
query tile 可以执行 fixed-16。Backward 不参与本次扫描。

运行 `compare.py`（单轮或多轮）和 `block_tuning.py` 均会在输出目录自动生成
`summary.md`，无需增加参数，CSV 输出继续保留。
`compare.py` 汇总各算子的型号、规格、策略、blocks、状态、TileLang/eager 耗时和 S；
`block_tuning.py` 按算子将 smoke/full 的各策略耗时、相对 original 收益及 S 并排展示，
VecAdd 包含固定 28、32 blocks。失败或缺失项不会显示为有效性能收益。

