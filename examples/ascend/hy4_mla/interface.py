"""BF16 FlashMLA-compatible entry point for the Hy4 preview H64 core.

Only metadata is inspected on the host. KV/indices must already use the same
physical row domain; causal/request masks and paged index conversion stay with
the caller. See ADAPTATION.md for supported strides and the small-head fallback.
"""

from contextlib import nullcontext
from functools import lru_cache
import math

import torch


def _check_tensor(tensor, name, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have last-dimension stride 1")
    # Allow pitch between non-overlapping rows.
    span = 1
    for extent, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if stride <= 0 or (extent > 1 and stride < span):
            raise ValueError(f"{name} requires positive, non-overlapping ordered strides")
        span += max(extent - 1, 0) * stride


def _overlapping_spans(a, b):
    if a.numel() == 0 or b.numel() == 0 or a.device != b.device:
        return False

    def interval(tensor):
        span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        begin = tensor.data_ptr()
        return begin, begin + span * tensor.element_size()

    a_begin, a_end = interval(a)
    b_begin, b_end = interval(b)
    return a_begin < b_end and b_begin < a_end


@lru_cache(maxsize=32)
def _get_kernel(nq, nk, topk, scale, sink, variable_lengths, q_strides, kv_stride, index_stride, out_strides, device):
    from examples.ascend.hy4_mla.compile_sparse_fwd import compile_bf16

    if torch.device(device).type != "npu":
        raise ValueError("Hy4 sparse MLA execution requires an Ascend NPU")
    return compile_bf16(
        nq,
        nk,
        topk,
        scale=scale,
        sink=sink,
        variable_lengths=variable_lengths,
        q_strides=q_strides,
        kv_stride=kv_stride,
        index_stride=index_stride,
        out_strides=out_strides,
        explicit_outputs=True,
    )


def flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None):
    """Return output/Max/LSE using the BF16 sparse FlashMLA calling convention.

    Q [N,H,576], KV [M,1,576], indices [N,1,K]; 1<=H<=64, K%64==0.
    Float32 sinks and int32 lengths are optional. Length values must be in
    [0,K]; no device-value validation or synchronization is performed here.
    Max/LSE exclude sink, and empty rows return (0,-inf,+inf).
    With H64, a supplied out is bound directly to the kernel. With H<64 an
    internal padded output is computed and copied into the supplied out.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must be a rank-3 tensor [N,H,576]")
    nq, heads, dim = q.shape
    if dim != 576 or not 1 <= heads <= 64:
        raise ValueError("q requires 1<=H<=64 and QK dimension 576")
    if d_v != 512:
        raise ValueError("d_v must be 512")
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be positive and finite")
    if not isinstance(kv, torch.Tensor) or kv.ndim != 3:
        raise ValueError("kv must have shape [M,1,576]")
    nk = kv.shape[0]
    if nq > 0 and nk == 0:
        raise ValueError("Nonempty queries require nonempty KV rows")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 3:
        raise ValueError("indices must have shape [N,1,K]")
    topk = indices.shape[-1]
    if topk <= 0 or topk % 64:
        raise ValueError("topk must be a positive multiple of 64")

    _check_tensor(q, "q", (nq, heads, 576), torch.bfloat16, q.device)
    _check_tensor(kv, "kv", (nk, 1, 576), torch.bfloat16, q.device)
    _check_tensor(indices, "indices", (nq, 1, topk), torch.int32, q.device)
    if attn_sink is not None:
        _check_tensor(attn_sink, "attn_sink", (heads,), torch.float32, q.device)
    if topk_length is not None:
        _check_tensor(topk_length, "topk_length", (nq,), torch.int32, q.device)
    if out is not None:
        _check_tensor(out, "out", (nq, heads, 512), torch.bfloat16, q.device)
        for name, tensor in (("q", q), ("kv", kv), ("indices", indices), ("attn_sink", attn_sink), ("topk_length", topk_length)):
            if tensor is not None and _overlapping_spans(out, tensor):
                raise ValueError(f"out storage span must not overlap {name}")

    maximum = torch.empty((nq, 64), dtype=torch.float32, device=q.device)
    lse = torch.empty_like(maximum)
    if out is None:
        out = torch.empty((nq, heads, 512), dtype=q.dtype, device=q.device)
    if nq == 0:
        return out, maximum[:, :heads], lse[:, :heads]

    use_sink = attn_sink is not None
    use_lengths = topk_length is not None
    if heads < 64:
        padded_q = q.new_zeros((nq, 64, 576))
        padded_q[:, :heads].copy_(q)
        q = padded_q
        if use_sink:
            padded_sink = attn_sink.new_full((64,), float("-inf"))
            padded_sink[:heads].copy_(attn_sink)
            attn_sink = padded_sink
        kernel_out = out.new_empty((nq, 64, 512))
    else:
        kernel_out = out
    # The disabled specializations leave these PrimFunc arguments unread.
    if not use_sink:
        attn_sink = torch.empty((64,), dtype=torch.float32, device=q.device)
    if not use_lengths:
        topk_length = torch.empty((nq,), dtype=torch.int32, device=q.device)
    flat_kv = kv.squeeze(1)
    flat_indices = indices.squeeze(1)
    with torch.npu.device(q.device) if q.device.type == "npu" else nullcontext():
        kernel = _get_kernel(
            nq,
            nk,
            topk,
            float(sm_scale),
            use_sink,
            use_lengths,
            tuple(q.stride()),
            flat_kv.stride(0),
            flat_indices.stride(0),
            tuple(kernel_out.stride()),
            str(q.device),
        )
        kernel(q, flat_kv, flat_indices, attn_sink, topk_length, kernel_out, maximum, lse)
    if heads < 64:
        out.copy_(kernel_out[:, :heads])
    return out, maximum[:, :heads], lse[:, :heads]
