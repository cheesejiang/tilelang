"""Explicit, pre-lowering block selection for the existing persistent examples."""

BLOCK_CASES = ("vecadd_simt", "vecadd_simd", "gemm_bf16", "gemm_fp32", "gemm_mixed_fp32")
POLICIES = ("original", "auto", "c-1", "c", "c+1", "2c")


def select_blocks(policy, original, target_cores, tasks):
    """Only auto clamps to task count; sweep candidates intentionally can oversubscribe."""
    if min(original, target_cores, tasks) <= 0:
        raise ValueError("Original blocks, target core count and task count must be positive")
    if policy == "original":
        return original
    if policy == "auto":
        return min(target_cores, tasks)
    choices = {"c-1": max(1, target_cores - 1), "c": target_cores,
               "c+1": target_cores + 1, "2c": 2 * target_cores}
    if policy not in choices:
        raise ValueError(f"Unknown block policy: {policy}")
    return choices[policy]
