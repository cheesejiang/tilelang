# Ascend hardware sensitivity experiments

This benchmark targets the **Ascend C++ / Bisheng backend**, explicitly using
`dav-3510` by default. It does not change TileLang's device selection or scheduler.
Run it on the 9579 board using an Ascend-enabled build of this checkout.

## Run

From the TileLang repository root, first run a small correctness/timing smoke test:

```bash
python examples/ascend/hardware_sensitivity/benchmark.py \
  --suite blocks --blocks 1 32 64 --block-working-set-mib 16 \
  --gemm-size 256 --samples 3 --repeats 3 \
  --no-plots --output hardware-smoke
```

Then run the complete experiments:

```bash
python examples/ascend/hardware_sensitivity/benchmark.py \
  --suite all --no-plots --output hardware-9579
```

Plotting requires `matplotlib`. Plots can be produced on a different host from
the CSV, without Torch, TileLang, or an NPU:

```bash
python examples/ascend/hardware_sensitivity/benchmark.py \
  --plot-only --output hardware-9579
```

Omit `--no-plots` to plot immediately after measurement if matplotlib is installed.
Each suite can also be selected individually with `--suite blocks`, `reuse`, or
`stream`. Use distinct output directories for distinct runs: results.csv is
overwritten on each measurement run.

## What is measured

### 1. Logical blocks versus performance (`blocks`)

- Copy and VecAdd use a fixed **256 MiB total input + output working set**.
  The default block sweep includes 1, 4, 8, 16, 28, available AIV count, 64,
  and twice available AIV count; duplicate values are removed.
- Persistent FP16 GEMM computes `A @ B.T` with FP32 output, default matrix size
  1024, tile size 64. Its sweep includes 1, 4, 8, 16, available Cube count,
  32, and twice available Cube count.
- Every block processes task IDs `bx, bx+B, bx+2B, ...`. Changing block count
  preserves total work and correctly handles a task count not divisible by B.
- These are simple sensitivity kernels, not claims of peak bandwidth or GEMM
  throughput. In particular the GEMM does not implement an optimized pipeline.

Use `--blocks ...` to override both sweeps, `--block-working-set-mib` to change
memory workload size, and `--gemm-size` (multiple of 64) to change GEMM size.

### 2. Working-set reuse (`reuse`)

Copy and VecAdd scan total working sets of 8, 16, 32, 64, 96, 128, 192, and
256 MiB, including output. Sizes are rounded down to whole tiles; **actual**
bytes are recorded. Each logical block repeatedly traverses its own tiles.
The number of rounds is chosen to reach at least `--reuse-traffic-mib` (default
512) of logical input/output traffic per invocation. Actual traffic and round
count are recorded, and can differ slightly between cases.

This is a repeated read/write reuse probe, not a read-only L2 test. A small
per-block working set can also benefit from caches other than shared L2. A
latency or bandwidth change does not by itself identify an exact L2 boundary.
Source code is saved for inspection; if investigating compiler elimination of
repeated accesses, inspect the compiled code/profiler traffic as well.

### 3. Single-pass streaming (`stream`)

The same working sets are scanned once per invocation. Sizes above the 9579's
128 MiB L2 are the relevant large-working-set cases, but cache policies and
residency still matter. Small cases can be cache-resident across launches:
**single-pass does not mean cold-cache**.

Reuse and stream use available AIV count by default; override with
`--memory-blocks`. Change the size sweep with `--working-sets-mib`.

## Timing and correctness

- Explicit output buffers are filled with NaNs before the correctness run.
  Copy/VecAdd are compared exactly; GEMM uses a Torch FP32 reference with
  rtol=atol=1e-3.
- Compilation, validation, allocations and reference computation are outside
  timing. Kernels are warmed up, then timed with Torch NPU events.
- Each sample averages `--repeats` consecutive launches. The report contains
  the median, min, max and all `--samples` batch averages, in microseconds.
  These are not individual-launch percentiles.
- There is **no explicit cache flush**. Host launch gaps can contribute to
  event intervals. Do not describe these numbers as profiler-isolated kernel
  instruction time. Run without competing device workloads.
- Experiment order is shuffled reproducibly with `--seed`. For confidence,
  run multiple sessions into separate directories and compare their variation.
- A compilation/correctness/device failure is recorded and stops the run;
  later cases are not attempted after a potentially asynchronous device error.

## Outputs and formulas

- `environment.json`: real device, core counts, memory capacity, package/import
  path, checkout commit, arguments, and available Bisheng/npu-smi version data.
- `results.csv`: one row per case, including samples and correctness status.
- `sources/*.asc`: generated sources.
- `blocks`, `reuse`, `stream` `.png`/`.svg`: latency and logical-bandwidth curves.
  GEMM TFLOP/s is in the CSV; it is not mixed onto a bandwidth axis.

```text
Copy logical bytes   = elements × 4 × 2 × rounds
VecAdd logical bytes = elements × 4 × 3 × rounds
Effective GB/s       = logical bytes / (median_us × 1000)
GEMM FLOPs           = 2 × size³
TFLOP/s              = FLOPs / (median_us × 10⁶)
```

Logical bandwidth is **not measured DRAM traffic/bandwidth**. Cache reuse can
make it exceed DRAM bandwidth. The unused second Copy argument aliases its
input and does not add a unique buffer to the working-set calculation.

## Scope of conclusions

Reducing blocks on a 9579 does **not** simulate a lower-core-count SKU: its
shared caches, memory bandwidth and frequency remain unchanged. Increasing
working set does not reconfigure its L2 capacity, and allocating less memory
does not simulate a different GM subsystem. These experiments identify
sensitivity and establish a baseline; they cannot establish another SKU's
performance or a before/after regression without measurements on that SKU.

## Real-operator follow-up: C-1, C, C+1, 2C

`real_operators.py` imports the existing `example_vecadd.vector_add` and
`example_gemm.gemm` implementations rather than the simple kernels above.
The examples now accept optional `num_blocks`, preserving defaults of 64 and
32 respectively. VecAdd also handles a final incomplete **block round**;
element counts must still be whole 8192-element tiles.

```bash
# Small first check (15 cases on 9579).
python examples/ascend/hardware_sensitivity/real_operators.py \
  --vector-elements 1048576 --gemm-sizes 1024 \
  --samples 3 --repeats 3 --no-plots --output real-smoke

# Full follow-up (25 cases on 9579).
python examples/ascend/hardware_sensitivity/real_operators.py \
  --no-plots --output real-9579
```

The default suite measures:

| Operator | Workload | C on 9579 | Block candidates |
|---|---|---:|---|
| Real SIMT VecAdd (pipelined) | 2^24 elements | 56 | 55, 56, 57, 112, original 64 |
| Real BF16 Cube GEMM | square 2048 and 4096, BF16 output | 28 | 27, 28, 29, 56, original 32 |
| Real BF16 mixed GEMM | square 2048 and 4096, FP32 output | 28 | 27, 28, 29, 56, original 32 |

Each group uses identical input values, shape, tiles, buffers and pipeline
configuration across block candidates, in shuffled order. Compilation and two
sentinel-output correctness checks are outside timing. Timing reuses the warm
NPU event method above. Reference buffers remain allocated throughout each
group, consistently across its block candidates. Cube and mixed GEMM have
different output types; compare block choices **within** a group rather than
attributing a direct cross-mode comparison solely to core type.

`results.csv` is flushed after each case. `summary.csv` adds
`latency_ratio_to_C` and `latency_ratio_to_original`: >1 means slower than that
baseline. Device information and generated sources are also saved. Plot with:

```bash
python examples/ascend/hardware_sensitivity/real_operators.py \
  --plot-only --output real-9579
```

Use `--operators vecadd`, `gemm_cube`, or `gemm_mixed` to isolate a workload.
RMSNorm and FlashAttention are not included: their current task partitioning
has divisibility constraints that prevent a fair same-shape C±1 sweep without
additional algorithm changes.
