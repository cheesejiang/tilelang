"""Explicit, pre-lowering block selection for the existing persistent examples."""

ATTENTION_CASES = ("mha_bf16", "gqa", "gqa_manual")
BLOCK_CASES = ("vecadd_simt", "vecadd_simd", "gemm_bf16", "gemm_fp32", "gemm_mixed_fp32") + ATTENTION_CASES
POLICIES = ("original", "auto", "c-1", "c", "c+1", "2c")
FIXED_POLICIES = ("fixed-16", "fixed-28", "fixed-32")
POLICIES = POLICIES + FIXED_POLICIES


def policies_for_case(case, profile=None):
    if case in ATTENTION_CASES:
        # Smoke MHA has only four query tiles; preserve its input/tile size.
        return ("original", "auto") if case == "mha_bf16" and profile == "smoke" else ("original", "auto", "fixed-16")
    if case.startswith("vecadd_"):
        return tuple(p for p in POLICIES if p != "fixed-16")
    return tuple(p for p in POLICIES if p not in FIXED_POLICIES)


def select_attention_blocks(policy, original, target_cores, tasks):
    """Keep the existing equal-work DSL: auto chooses the largest legal divisor."""
    blocks = select_blocks(policy, original, target_cores, tasks)
    if policy == "auto":
        while tasks % blocks:
            blocks -= 1
    if tasks % blocks:
        raise ValueError(f"{tasks} query tiles must be divisible by {blocks} blocks")
    return blocks


def select_blocks(policy, original, target_cores, tasks):
    """Only auto clamps to task count; sweep candidates intentionally can oversubscribe."""
    if min(original, target_cores, tasks) <= 0:
        raise ValueError("Original blocks, target core count and task count must be positive")
    if policy == "original":
        return original
    if policy == "auto":
        return min(target_cores, tasks)
    choices = {"fixed-16": 16, "fixed-28": 28, "fixed-32": 32,
               "c-1": max(1, target_cores - 1), "c": target_cores,
               "c+1": target_cores + 1, "2c": 2 * target_cores}
    if policy not in choices:
        raise ValueError(f"Unknown block policy: {policy}")
    return choices[policy]
