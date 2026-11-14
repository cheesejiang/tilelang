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
