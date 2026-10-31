"""Source-backed workload descriptions; importing this file needs no NPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    category: str
    source: str
    symbol: str
    adaptation: str


CASES = {
    "gemm_scale": Case("cross_single_pass", "examples/ascend/deepgemm/kernels/epilogue.py", "apply_alpha",
                       "FP32 scaling stage rewritten from SIMD to column-major SIMT; omit BF16 packing and GEMM."),
    "attention_finalize": Case("cross_single_pass", "examples/ascend/flash_attention/core.py", "finalize_fp32",
                               "O/l stage rewritten from SIMD to column-major SIMT; supply positive l as input; omit attention."),
    "attention_update": Case("cross_reused", "examples/ascend/flash_attention/core.py", "accumulate_output + kv_auto",
                             "Keep O across real, different incoming KV tiles; supply alpha as input; omit QK/PV GEMMs."),
    "attention_update_finalize": Case("cross_reused", "examples/ascend/flash_attention/core.py", "accumulate_output + finalize_fp32",
                                      "Same KV accumulation, followed by O/l; SIMD stages rewritten to SIMT."),
    "vecadd": Case("continuous_ownership", "examples/ascend/example_vecadd.py", "vector_add(mode='simt')",
                   "Keep A+B; use one tile per block, no pipeline/multiversion; same unannotated IR in all modes."),
    "gemm_bias": Case("continuous_ownership", "examples/ascend/deepgemm/kernels/epilogue.py", "add_bias",
                      "FP32 bias epilogue rewritten from SIMD to contiguous SIMT; omit BF16 conversion and GEMM."),
}

MODES = ("baseline", "ownership", "padding", "auto")


def mode_configs(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return {"tl.ascend_layout_diagnostics": True,
            "tl.ascend_enable_ub_ownership": mode in ("ownership", "auto"),
            "tl.ascend_enable_ub_padding": mode in ("padding", "auto")}


def validate(rows, columns, threads, blocks, updates):
    if min(rows, columns, threads, blocks, updates) <= 0:
        raise ValueError("all dimensions, blocks and updates must be positive")
    if threads % 32 or rows % threads:
        raise ValueError("threads must be a multiple of 32, and rows must be divisible by threads")
    if rows % 8:
        raise ValueError("rows must be a multiple of 8")
    # Work and incoming coexist; leave room for row statistics and allocator state.
    if (2 * rows * columns + 2 * rows + columns) * 4 > 208 * 1024:
        raise ValueError("tile exceeds the conservative example UB budget; reduce rows/columns")
