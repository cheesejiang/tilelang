"""Explicit, pre-lowering block selection for the existing persistent examples."""

BLOCK_CASES = ("vecadd_simt", "vecadd_simd", "gemm_bf16", "gemm_fp32", "gemm_mixed_fp32")
POLICIES = ("original", "auto", "c-1", "c", "c+1", "2c")
FIXED_POLICIES = ("fixed-28", "fixed-32")
POLICIES = POLICIES + FIXED_POLICIES


def policies_for_case(case):
    return POLICIES if case.startswith("vecadd_") else tuple(p for p in POLICIES if p not in FIXED_POLICIES)


def select_blocks(policy, original, target_cores, tasks):
    """Only auto clamps to task count; sweep candidates intentionally can oversubscribe."""
    if min(original, target_cores, tasks) <= 0:
        raise ValueError("Original blocks, target core count and task count must be positive")
    if policy == "original":
        return original
    if policy == "auto":
        return min(target_cores, tasks)
    choices = {"fixed-28": 28, "fixed-32": 32,
               "c-1": max(1, target_cores - 1), "c": target_cores,
               "c+1": target_cores + 1, "2c": 2 * target_cores}
    if policy not in choices:
        raise ValueError(f"Unknown block policy: {policy}")
    return choices[policy]
