"""CPU equations; work with either PyTorch tensors or NumPy arrays."""


def reference(case, a, incoming, stats):
    if case == "flash_decode_qk_column_store":
        # Quantized BF16 Q/K are converted back to FP32 before the reference
        # dot product, matching the device's FP32 accumulation input values.
        q = a.float() if hasattr(a, "float") else a.astype("float32")
        k = incoming.float() if hasattr(incoming, "float") else incoming.astype("float32")
        logits = q[:, None] @ k.swapaxes(-1, -2)
        maxima = logits.amax(dim=-1) if hasattr(logits, "amax") else logits.max(axis=-1)
        return maxima.swapaxes(1, 2)
    if case == "flash_decode_column_store":
        maxima = incoming.amax(dim=-1) if hasattr(incoming, "amax") else incoming.max(axis=-1)
        return maxima.swapaxes(1, 2)
    if case in ("kda_column_store", "kda_reduce_column_store"):
        result = a * 1.0
        result[:, :, 0] = stats[:, 0, :] if case == "kda_column_store" else (a * incoming[:, 0]).sum(-1)
        return result
    if case == "gemm_scale":
        return a * 1.25
    if case == "attention_finalize":
        return a / stats[:, -1, :, None]
    if case == "vecadd":
        return a + incoming[:, 0]
    if case == "gemm_bias":
        return a + incoming[:, 0, 0:1, :]
    if case not in ("attention_update", "attention_update_finalize"):
        raise ValueError(f"Unknown workload: {case}")
    result = a * 1.0
    for kv in range(incoming.shape[1]):
        result = result * stats[:, kv, :, None] + incoming[:, kv]
    return result / stats[:, -1, :, None] if case == "attention_update_finalize" else result
