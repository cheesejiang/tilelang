"""Fused rotate-half RoPE forward and input gradient on Ascend.

Run the original workload with `python example_fused_rope.py`, or search
SIMD tile sizes with `--tune`.
"""

import argparse
from functools import partial

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.carver.arch.driver.ascend_driver import get_num_vector_cores
from tilelang.profiler import do_bench

FLOAT_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
UB_BYTES = 208 * 1024
FP32_LANES = 64
S = T.simd


def tile_limit(heads, dim, dtype, cos_dtype, sin_dtype, position_dtype=torch.int64):
    data_size, cos_size, sin_size, position_size = (
        torch.empty((), dtype=t).element_size() for t in (dtype, cos_dtype, sin_dtype, position_dtype)
    )
    data_half = aligned_half(dim, data_size)
    cos_half = aligned_half(dim, cos_size)
    sin_half = aligned_half(dim, sin_size)
    # Reserve space for two versions of each buffer, their alignment, and vector overreads.
    reserve = 12 * 32 + 2 * FP32_LANES * (data_size + cos_size + sin_size)
    data_bytes = 8 * heads * data_half * data_size
    table_bytes = 4 * (cos_half * cos_size + sin_half * sin_size) + 2 * position_size
    return (UB_BYTES - reserve) // (data_bytes + table_bytes)


def aligned_half(dim, element_size):
    alignment = 32 // element_size
    return tilelang.cdiv(dim // 2, alignment) * alignment


def fused_rope(
    total_k,
    max_seq,
    num_cores,
    heads,
    dim,
    dtype,
    cos_dtype,
    sin_dtype,
    position_dtype,
    tile_tokens=None,
    backward=False,
):
    """Build a specialization for flattened token batches and nonnegative positions."""
    if min(total_k, max_seq, num_cores, heads, dim) <= 0 or dim % 2:
        raise ValueError("Expected positive sizes and an even head dimension")
    supported = FLOAT_DTYPES.values()
    if any(t not in supported for t in (dtype, cos_dtype, sin_dtype)):
        raise TypeError("RoPE data and caches support BF16, FP16 and FP32")
    if position_dtype not in (torch.int32, torch.int64):
        raise TypeError("Positions must be int32 or int64")
    limit = tile_limit(heads, dim, dtype, cos_dtype, sin_dtype, position_dtype)
    if limit == 0:
        raise ValueError("SIMD requires one token fitting in UB")
    if tile_tokens is None:
        tile_tokens = min(tilelang.cdiv(total_k, num_cores), max(1, (limit + 1) // 2))
    if not 1 <= tile_tokens <= limit:
        raise ValueError(f"tile_tokens must be between 1 and {limit}")
    data_half, cos_half, sin_half = (
        aligned_half(dim, torch.empty((), dtype=t).element_size()) for t in (dtype, cos_dtype, sin_dtype)
    )
    dtype, cos_dtype, sin_dtype, position_dtype = (
        str(t).removeprefix("torch.") for t in (dtype, cos_dtype, sin_dtype, position_dtype)
    )
    half = dim // 2
    blocks = min(num_cores, tilelang.cdiv(total_k, tile_tokens))
    groups, tail_tiles = divmod(tilelang.cdiv(total_k, tile_tokens), blocks)

    tile_rows = tile_tokens * heads * 2
    tile_elems = tile_rows * half
    coeff_rows = tile_tokens * 2
    chunks = tilelang.cdiv(half, FP32_LANES)

    @T.macro
    def load_f32(buf, offset, storage_dtype, mask, value):
        if storage_dtype == "float32":
            value[0] = S.vld(buf[offset])
        else:
            raw = S.vld(buf[offset], "UNPK_B16")
            value[0] = S.vcvt(raw, "float32", mask, part=0)

    @T.macro
    def rotate_heads(x, y, index, c, signed_s, partner, mask, lanes):
        values = T.alloc_local((1,), "float32x64")
        load_f32(x, index, dtype, mask, values)
        rotated = S.vselr(values[0], partner)
        result = S.vadd(S.vmul(values[0], c, mask), S.vmul(rotated, signed_s, mask), mask)
        if dtype == "float32":
            S.vsts(y[index], result, mask, extent=lanes)
        else:
            S.vsts(y[index], S.vcvt(result, dtype, mask, sat=False), mask, "PK_B32", extent=lanes)

    @T.macro
    def rotate_dim32(x, y, cos, sin, valid):
        full = S.pset(32)
        one_head = S.update_mask(dim)
        # CANN's arange provides signed integer registers. vselr reinterprets
        # these nonnegative indices as unsigned before selecting lanes.
        lane = S.vci(0, "int32")
        coeff_index = S.vand(lane, S.vdup(dim - 1, "int32"), full)
        partner = S.vxor(lane, S.vdup(half, "int32"), full)
        low_half = S.vcmps(coeff_index, half, full, op="lt")
        for token in range(valid):
            c_raw = T.alloc_local((1,), "float32x64")
            s_raw = T.alloc_local((1,), "float32x64")
            load_f32(cos, token * dim, cos_dtype, one_head, c_raw)
            load_f32(sin, token * dim, sin_dtype, one_head, s_raw)
            c = S.vselr(c_raw[0], coeff_index)
            s = S.vselr(s_raw[0], coeff_index)
            # Each lane keeps its original output position. Swapping bit 4 of
            # the lane index exchanges the halves inside each 32-element head.
            if backward:
                swapped_s = S.vselr(s, partner)
                signed_s = S.vsel(swapped_s, S.vneg(swapped_s, full), low_half)
            else:
                signed_s = S.vsel(S.vneg(s, full), s, low_half)
            for head_pair in range(heads // 2):
                index = (token * heads + head_pair * 2) * dim
                rotate_heads(x, y, index, c, signed_s, partner, full, FP32_LANES)
            if heads % 2:
                tail_index = (token * heads + heads - 1) * dim
                rotate_heads(x, y, tail_index, c, signed_s, partner, one_head, dim)

    @T.macro
    def rotate_simd_tile(X, Pos, Cos, Sin, Y, x, y, cos, sin, pos_ub, begin, valid):
        # Pad each half independently so aligned vector accesses can start at either half.
        first_row = begin * heads * 2
        if data_half == half:
            first_elem = first_row * half
            T.copy(X[first_elem : first_elem + tile_elems], x[:tile_elems])
        else:
            x_rows = T.Tensor((tile_rows, data_half), dtype, x.data)
            T.copy(X[first_row : first_row + tile_rows, :], x_rows[:, :half], pad_value=0)
        T.copy(Pos[begin : begin + tile_tokens], pos_ub[:tile_tokens])

        if cos_half != half:
            cos_rows = T.Tensor((coeff_rows, cos_half), cos_dtype, cos.data)
        if sin_half != half:
            sin_rows = T.Tensor((coeff_rows, sin_half), sin_dtype, sin.data)
        for cache_token in T.serial(valid):
            pos = T.Cast("int32", T.min(pos_ub[cache_token], T.Cast(position_dtype, max_seq - 1)))
            if cos_half == half:
                T.copy(Cos[pos, :], cos[cache_token * dim : (cache_token + 1) * dim])
            else:
                T.copy(Cos[2 * pos : 2 * pos + 2, :], cos_rows[2 * cache_token : 2 * cache_token + 2, :half], pad_value=0)
            if sin_half == half:
                T.copy(Sin[pos, :], sin[cache_token * dim : (cache_token + 1) * dim])
            else:
                T.copy(Sin[2 * pos : 2 * pos + 2, :], sin_rows[2 * cache_token : 2 * cache_token + 2, :half], pad_value=0)

        with T.SimdVF():
            if dim * 2 == FP32_LANES:
                rotate_dim32(x, y, cos, sin, valid)
            else:
                for token in range(valid):
                    for chunk in range(chunks):
                        lanes = T.min(FP32_LANES, half - chunk * FP32_LANES)
                        mask = S.update_mask(lanes)
                        cos_index = token * 2 * cos_half + chunk * FP32_LANES
                        sin_index = token * 2 * sin_half + chunk * FP32_LANES
                        c_lo = T.alloc_local((1,), "float32x64")
                        c_hi = T.alloc_local((1,), "float32x64")
                        s_lo = T.alloc_local((1,), "float32x64")
                        s_hi = T.alloc_local((1,), "float32x64")
                        load_f32(cos, cos_index, cos_dtype, mask, c_lo)
                        load_f32(cos, cos_index + cos_half, cos_dtype, mask, c_hi)
                        load_f32(sin, sin_index, sin_dtype, mask, s_lo)
                        load_f32(sin, sin_index + sin_half, sin_dtype, mask, s_hi)
                        for head in range(heads):
                            index = (token * heads + head) * 2 * data_half + chunk * FP32_LANES
                            lo = T.alloc_local((1,), "float32x64")
                            hi = T.alloc_local((1,), "float32x64")
                            load_f32(x, index, dtype, mask, lo)
                            load_f32(x, index + data_half, dtype, mask, hi)
                            diag_lo = S.vmul(lo[0], c_lo[0], mask)
                            diag_hi = S.vmul(hi[0], c_hi[0], mask)
                            if backward:
                                out_lo = S.vadd(diag_lo, S.vmul(hi[0], s_hi[0], mask), mask)
                                out_hi = S.vsub(diag_hi, S.vmul(lo[0], s_lo[0], mask), mask)
                            else:
                                out_lo = S.vsub(diag_lo, S.vmul(hi[0], s_lo[0], mask), mask)
                                out_hi = S.vadd(diag_hi, S.vmul(lo[0], s_hi[0], mask), mask)
                            if dtype == "float32":
                                S.vsts(y[index], out_lo, mask, extent=lanes)
                                S.vsts(y[index + data_half], out_hi, mask, extent=lanes)
                            else:
                                S.vsts(y[index], S.vcvt(out_lo, dtype, mask, sat=False), mask, "PK_B32", extent=lanes)
                                S.vsts(y[index + data_half], S.vcvt(out_hi, dtype, mask, sat=False), mask, "PK_B32", extent=lanes)
        if data_half == half:
            T.copy(y[:tile_elems], Y[first_elem : first_elem + tile_elems])
        else:
            y_rows = T.Tensor((tile_rows, data_half), dtype, y.data)
            T.copy(y_rows[:, :half], Y[first_row : first_row + tile_rows, :])

    @T.prim_func
    def main(
        X: T.Tensor((total_k, heads, dim), dtype),
        Pos: T.Tensor((total_k,), position_dtype),
        Cos: T.Tensor((max_seq, dim), cos_dtype),
        Sin: T.Tensor((max_seq, dim), sin_dtype),
        Y: T.Tensor((total_k, heads, dim), dtype),
    ):
        with T.Kernel(blocks) as core:
            if data_half == half:
                x_view = T.Tensor((total_k * heads * dim,), dtype, X.data)
                y_view = T.Tensor((total_k * heads * dim,), dtype, Y.data)
            else:
                x_view = T.Tensor((total_k * heads * 2, half), dtype, X.data)
                y_view = T.Tensor((total_k * heads * 2, half), dtype, Y.data)
            if cos_half == half:
                cos_view = Cos
            else:
                cos_view = T.Tensor((max_seq * 2, half), cos_dtype, Cos.data)
            if sin_half == half:
                sin_view = Sin
            else:
                sin_view = T.Tensor((max_seq * 2, half), sin_dtype, Sin.data)
            x = T.alloc_shared((tile_rows * data_half + FP32_LANES,), dtype)
            y = T.alloc_shared((tile_rows * data_half,), dtype)
            cos = T.alloc_shared((coeff_rows * cos_half + FP32_LANES,), cos_dtype)
            sin = T.alloc_shared((coeff_rows * sin_half + FP32_LANES,), sin_dtype)
            pos_ub = T.alloc_shared((tile_tokens,), position_dtype)
            iterations = groups + T.Cast("int32", core < tail_tiles)
            for step in T.Pipelined(iterations, num_stages=2):
                begin = (step * blocks + core) * tile_tokens
                valid = T.min(tile_tokens, total_k - begin)
                rotate_simd_tile(x_view, Pos, cos_view, sin_view, y_view, x, y, cos, sin, pos_ub, begin, valid)

    return main


def compile_rope(x, positions, cos, sin, *, backward=False, num_cores=None, tile_tokens=None):
    """Compile a kernel for contiguous flattened views of the supplied tensors."""
    if x.ndim < 2:
        raise ValueError("Input must end in (heads, head_dim)")
    heads, dim = x.shape[-2:]
    if min(heads, dim) <= 0 or cos.ndim != 2 or cos.shape != sin.shape or cos.shape[1] != dim:
        raise ValueError("cos and sin must have shape (max_seq, head_dim)")
    tokens = x.numel() // (heads * dim)
    if positions.numel() != tokens:
        raise ValueError("Expected one position per token")
    if any(t.device != x.device for t in (positions, cos, sin)):
        raise ValueError("All RoPE inputs must be on the same device")
    if num_cores is None:
        num_cores = get_num_vector_cores(x.device.index or 0)
    return tilelang.compile(
        fused_rope(
            tokens,
            cos.shape[0],
            num_cores,
            heads,
            dim,
            x.dtype,
            cos.dtype,
            sin.dtype,
            positions.dtype,
            tile_tokens=tile_tokens,
            backward=backward,
        ),
        target="ascend",
    )


def launch(x, positions, cos, sin, *, backward=False, **config):
    """Apply RoPE or its input gradient with shape (..., heads, head_dim)."""
    if x.ndim >= 3 and 0 in x.shape[:-2] and positions.numel() == 0:
        return torch.empty_like(x)
    kernel = compile_rope(x, positions, cos, sin, backward=backward, **config)
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    kernel(
        x.contiguous().view(-1, *x.shape[-2:]),
        positions.contiguous().view(-1),
        cos.contiguous(),
        sin.contiguous(),
        output.view(-1, *x.shape[-2:]),
    )
    return output


def reference(x, positions, cos, sin, backward=False):
    positions = positions.clamp(max=cos.shape[0] - 1)
    cos = cos[positions, None].float()
    sin = sin[positions, None].float()
    lo, hi = x.float().chunk(2, dim=-1)
    c_lo, c_hi = cos.chunk(2, dim=-1)
    s_lo, s_hi = sin.chunk(2, dim=-1)
    if backward:
        return torch.cat((lo * c_lo + hi * s_hi, hi * c_hi - lo * s_lo), dim=-1).to(x.dtype)
    return torch.cat((lo * c_lo - hi * s_lo, hi * c_hi + lo * s_hi), dim=-1).to(x.dtype)


def check_accuracy(output, x, positions, cos, sin, backward):
    # Keep peak CPU memory bounded even at 600,000 tokens.
    squared_error = squared_reference = max_error = 0.0
    tolerance = 1e-6 if x.dtype == torch.float32 else 1e-2
    for begin in range(0, x.shape[0], 4096):
        chunk = slice(begin, begin + 4096)
        expected = reference(x[chunk].cpu(), positions[chunk], cos, sin, backward)
        actual = output[chunk].cpu()
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        error = actual.float() - expected.float()
        max_error = max(max_error, error.abs().max().item())
        squared_error += error.square().sum().item()
        squared_reference += expected.float().square().sum().item()
    return max_error, (squared_error / max(squared_reference, 1e-12)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-k", type=int, default=600_000)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=FLOAT_DTYPES, default="bfloat16")
    parser.add_argument("--cache-dtype", choices=FLOAT_DTYPES)
    parser.add_argument("--cores", type=int, help="override the device vector core count")
    parser.add_argument("--tile-tokens", type=int)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if min(args.total_k, args.max_seq, args.iters, args.heads, args.head_dim) <= 0 or args.head_dim % 2:
        parser.error("sizes and iterations must be positive, and head-dim must be even")
    if args.cores is not None and args.cores <= 0:
        parser.error("cores must be positive")
    dtype = FLOAT_DTYPES[args.dtype]
    cache_dtype = FLOAT_DTYPES[args.cache_dtype or args.dtype]
    limit = tile_limit(args.heads, args.head_dim, dtype, cache_dtype, cache_dtype)
    if limit == 0:
        parser.error("SIMD requires one token fitting in UB")
    if args.tile_tokens is not None and not 1 <= args.tile_tokens <= limit:
        parser.error(f"tile-tokens must be between 1 and {limit}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("An Ascend NPU and torch_npu are required")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    positions = torch.arange(args.total_k, dtype=torch.int64) % args.max_seq
    positions[-1] = args.max_seq  # Exercise the upper clamp.
    freq = 10000.0 ** (-torch.arange(0, args.head_dim, 2, dtype=torch.float32) / args.head_dim)
    angles = torch.outer(torch.arange(args.max_seq, dtype=torch.float32), freq).repeat(1, 2)
    cos, sin = angles.cos().to(cache_dtype), angles.sin().to(cache_dtype)
    tables = tuple(t.to(device) for t in (positions, cos, sin))
    shape = (args.total_k, args.heads, args.head_dim)
    output = torch.empty(shape, device=device, dtype=dtype)
    num_cores = args.cores if args.cores is not None else get_num_vector_cores(device.index or 0)
    tokens_per_core = tilelang.cdiv(args.total_k, num_cores)
    default_tile = min(tokens_per_core, (limit + 1) // 2)
    if args.tune and args.tile_tokens is None:
        tiles = sorted({min(tokens_per_core, max(1, tile)) for tile in (limit // 4, default_tile, 3 * limit // 4, limit)})
    else:
        tiles = [args.tile_tokens or default_tile]
    bench = partial(
        do_bench, backend="event", device=device, return_mode="median", _n_warmup=10, _n_repeat=args.iters
    )
    io_bytes = 2 * output.numel() * output.element_size()

    for backward in (False, True):
        label = "backward" if backward else "forward"
        x = torch.randn(shape, device=device, dtype=dtype)
        if not backward:
            # Reuse the preallocated RoPE buffers for the same-device copy.
            copy_ms = bench(lambda: output.copy_(x))
            if copy_ms <= 0:
                raise RuntimeError("Device timer did not resolve the copy baseline")
            print(f"copy baseline (torch.Tensor.copy_)  {copy_ms:.4f} ms  {io_bytes / (copy_ms * 1e6):.1f} GB/s")
        timings = []
        for tile_tokens in tiles:
            kernel = compile_rope(x, *tables, num_cores=num_cores, tile_tokens=tile_tokens, backward=backward)
            kernel(x, *tables, output)
            max_error, rel_l2 = check_accuracy(output, x, positions, cos, sin, backward)
            latency = bench(lambda: kernel(x, *tables, output))
            gbps = io_bytes / (latency * 1e6)
            timings.append((latency, tile_tokens))
            print(
                f"{label:8} simd tile={tile_tokens:2}  {latency:.4f} ms  {gbps:.1f} GB/s (input/output)  "
                f"copy_bw={copy_ms / latency:.1%}  max_abs={max_error:.3e}  rel_l2={rel_l2:.3e}"
            )
        if args.tune:
            latency, tile_tokens = min(timings)
            print(f"{label:8} simd best tile={tile_tokens:2}  {latency:.4f} ms  copy_bw={copy_ms / latency:.1%}")
        del x
    print("ACCURACY: PASS")


if __name__ == "__main__":
    main()
