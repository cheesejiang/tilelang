# Ascend compile-time block allocation

The plain `ascend` backend runs `tl.ascend.AutoBlocks` immediately after
materializing the launch, before task scheduling, synchronization insertion,
and host/device splitting. Existing operator DSLs do not need a new argument.

For an eligible independent task loop, the policy is:

```text
blocks = min(resource core count, task tile count)
```

A kernel containing Cube GEMM uses the AIC budget, including an automatically
scheduled Cube/Vector epilogue. A pure SIMD/SIMT VF kernel uses the AIV budget.
Unrecognized execution modes are left unchanged.

The selected current NPU is queried first. A missing resource count falls back
to `TILELANG_ASCEND_AIC_CORES` or `TILELANG_ASCEND_AIV_CORES`. These must be positive
integers. With no device count and no manual count, that resource is not optimized.
The assumption is that compilation and execution use the current device.

```bash
# Manual fallback when compiling without an NPU; a detected device takes priority.
TILELANG_ASCEND_AIC_CORES=28 TILELANG_ASCEND_AIV_CORES=56 python your_program.py

# Preserve original grids for diagnostics or before/after measurements.
TILELANG_ASCEND_AUTO_BLOCKS=0 python your_program.py
```

Core budgets are frozen into PrimFunc attributes before full-kernel cache lookup.
JIT in-memory caches are invalidated if this policy's device budgets change.
Direct staged lowering resolves the same policy when entering the pass.

## Eligibility

The pass recognizes one-dimensional `T.Persistent` task domains and canonical
grid-stride loops whose complete task index is `step * original_blocks + bx`.
It changes the task index, iteration count, and grid together. If a previously
exact grid gains a partial last round, the complete task body is guarded.

Rewriting requires a recognized output ownership pattern, no global-memory
feedback, and scratch state initialized within each task. Unannotated vector
loops must write nonoverlapping one-dimensional task regions. The pass skips
explicit synchronization, atomic operations, multidimensional/explicit mixed
launches, ambiguous loop structures, and unknown task state.

Contiguous equal-work schedules such as the current Attention forward are not
converted. Fixed grids without a recognized task loop, dynamic task bounds,
and DeepGEMM's separate scheduler are retained. There is no divisor search,
offline tuning, or performance model, and no guarantee of optimal performance.

The eager block-sweep driver explicitly disables this production pass on its
candidate PrimFuncs, preserving the meaning of `original` and explicit block
experiments.

## Validation

Host-only IR tests (with an Ascend-enabled TileLang build):

```bash
python -m pytest testing/ascend/transform/test_ascend_auto_blocks.py -q
```

On an NPU, the original correctness entry points exercise the production pass:

```bash
TILELANG_DEFAULT_TARGET=ascend TILELANG_EXECUTION_BACKEND=tvm_ffi \
  python -m pytest examples/ascend/test_vecadd.py examples/ascend/test_gemm.py -v
```

Use `TILELANG_ASCEND_AUTO_BLOCKS=0` for the original compiler behavior when
comparing before/after. Host-only IR coverage and generated source checks do
not replace numerical validation on the NPU.
