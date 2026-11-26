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

TOTAL_K = 600_000
HEADS = 56
HEAD_DIM = 32
MAX_SEQ = 4096
FP16 = "float16"
FP16_BYTES = 2
FP16_LANES = 128
UB_BYTES = 208 * 1024
TOKEN_ELEMS = HEADS * HEAD_DIM
CACHE_ROW = 2 * HEAD_DIM
S = T.simd


def tile_limit():
    # Two versions of input, output, packed coefficients, and positions.
    coeff_guard = FP16_LANES - CACHE_ROW
    reserve = 8 * 32 + 2 * coeff_guard * FP16_BYTES
    bytes_per_token = 4 * (TOKEN_ELEMS + HEAD_DIM) * FP16_BYTES + 2 * 8
    return (UB_BYTES - reserve) // bytes_per_token


def tile_grid(total_k, num_cores, tile_tokens):
    num_tiles = tilelang.cdiv(total_k, tile_tokens)
    blocks = min(num_cores, num_tiles)
    groups, tail_tiles = divmod(num_tiles, blocks)
    return blocks, groups, tail_tiles


def mte2_read_baseline(total_k, num_cores, tile_tokens):
    """Stream RoPE input DMA tiles and write one 32-byte sink per core."""
    sink_width = 32 // FP16_BYTES
    total_elems = total_k * TOKEN_ELEMS
    tile_elems = tile_tokens * TOKEN_ELEMS
    blocks, groups, tail_tiles = tile_grid(total_k, num_cores, tile_tokens)

    @T.prim_func
    def main(X: T.Tensor((total_k, HEADS, HEAD_DIM), FP16), Sink: T.Tensor((blocks, sink_width), FP16)):
        with T.Kernel(blocks) as core:
            src = T.Tensor((total_elems,), FP16, X.data)
            ub = T.alloc_shared((tile_elems,), FP16)
            iterations = groups + T.Cast("int32", core < tail_tiles)
            for step in T.serial(iterations):
                start = (step * blocks + core) * tile_elems
                T.copy(src[start : start + tile_elems], ub)
            T.copy(ub[:sink_width], Sink[core, :])

    return main


def bench_mte2_read(x, num_cores, tile_tokens, bench):
    total_k = x.shape[0]
    blocks, _, _ = tile_grid(total_k, num_cores, tile_tokens)
    sink = x.new_empty((blocks, 32 // FP16_BYTES))
    kernel = tilelang.compile(mte2_read_baseline(total_k, num_cores, tile_tokens), target="ascend")
    kernel(x, sink)
    core = torch.arange(blocks, device=x.device)
    last_tile = core + ((tilelang.cdiv(total_k, tile_tokens) - 1 - core) // blocks) * blocks
    torch.testing.assert_close(sink[:, 0], x[last_tile * tile_tokens, 0, 0], rtol=0, atol=0)
    latency = bench(lambda: kernel(x, sink))
    if latency <= 0:
        raise RuntimeError("Device timer did not resolve the GM-to-UB baseline")
    return latency, x.numel() * FP16_BYTES / (latency * 1e6)


def dma_copy_baseline(total_k, num_cores, tile_tokens, *, offset=False):
    """Use the same tile assignment for a double-buffered device copy."""
    total_elems = total_k * TOKEN_ELEMS
    tile_elems = tile_tokens * TOKEN_ELEMS
    blocks, groups, tail_tiles = tile_grid(total_k, num_cores, tile_tokens)
    schedule = {"enable_offset": True} if offset else {}

    @T.prim_func
    def main(X: T.Tensor((total_k, HEADS, HEAD_DIM), FP16), Y: T.Tensor((total_k, HEADS, HEAD_DIM), FP16)):
        with T.Kernel(blocks) as core:
            src = T.Tensor((total_elems,), FP16, X.data)
            dst = T.Tensor((total_elems,), FP16, Y.data)
            ub = T.alloc_shared((tile_elems,), FP16)
            T.annotate_buffer_versions({ub: 2})
            iterations = groups + T.Cast("int32", core < tail_tiles)
            for step in T.Pipelined(iterations, num_stages=2, annotations=schedule):
                start = (step * blocks + core) * tile_elems
                if offset:
                    with T.Stage(0):
                        T.copy(src[start : start + tile_elems], ub)
                    with T.Stage(1):
                        T.copy(ub, dst[start : start + tile_elems])
                else:
                    T.copy(src[start : start + tile_elems], ub)
                    T.copy(ub, dst[start : start + tile_elems])

    return main


def bench_dma_copy(x, output, num_cores, tile_tokens, bench, *, offset=False):
    kernel = tilelang.compile(dma_copy_baseline(x.shape[0], num_cores, tile_tokens, offset=offset), target="ascend")
    output.fill_(float("nan"))
    kernel(x, output)
    torch.testing.assert_close(output[::tile_tokens, 0, 0], x[::tile_tokens, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(output[-1], x[-1], rtol=0, atol=0)
    latency = bench(lambda: kernel(x, output))
    if latency <= 0:
        raise RuntimeError("Device timer did not resolve the TileLang DMA copy baseline")
    return latency


def fused_rope(total_k, num_cores, tile_tokens=None, backward=False, prefetch=False, position_prefetch=False):
    """Build the 56-head, 32-element FP16 kernel."""
    limit = tile_limit()
    if min(total_k, num_cores) <= 0:
        raise ValueError("Expected positive token and core counts")
    if tile_tokens is None:
        tile_tokens = min(tilelang.cdiv(total_k, num_cores), (limit + 1) // 2)
    if not 1 <= tile_tokens <= limit:
        raise ValueError(f"tile_tokens must be between 1 and {limit}")
    blocks, groups, tail_tiles = tile_grid(total_k, num_cores, tile_tokens)
    total_elems = total_k * TOKEN_ELEMS
    tile_elems = tile_tokens * TOKEN_ELEMS
    coeff_elems = tile_tokens * CACHE_ROW
    coeff_guard = FP16_LANES - CACHE_ROW
    prefetch = prefetch or position_prefetch
    schedule = {"enable_offset": True} if prefetch else {}

    @T.macro
    def load_coefficients(Cache, pos_ub, coeff_ub, valid):
        for i in T.serial(valid):
            pos = T.Cast("int32", T.min(pos_ub[i], T.int64(MAX_SEQ - 1)))
            T.copy(Cache[pos, :], coeff_ub[i * CACHE_ROW : (i + 1) * CACHE_ROW])

    @T.macro
    def rotate(x, y, coeff_ub, valid):
        with T.SimdVF():
            mask = S.pset(16)
            lane = S.vci(0, "int16")
            coeff_index = S.vand(lane, S.vdup(HEAD_DIM - 1, "int16"), mask)
            partner = S.vxor(lane, S.vdup(HEAD_DIM // 2, "int16"), mask)
            low_half = S.vcmps(coeff_index, HEAD_DIM // 2, mask, op="lt")
            # Flip the FP16 sign bit without widening the coefficient.
            sign_bit = S.vdup(0x8000, "uint16")
            zero = S.vdup(0, "uint16")
            if backward:
                sin_index = S.vxor(coeff_index, S.vdup(HEAD_DIM + HEAD_DIM // 2, "int16"), mask)
                sign = S.vsel(zero, sign_bit, low_half)
            else:
                sin_index = S.vxor(coeff_index, S.vdup(HEAD_DIM, "int16"), mask)
                sign = S.vsel(sign_bit, zero, low_half)
            for token in range(valid):
                coeff_bits = T.reinterpret(S.vld(coeff_ub[token * CACHE_ROW]), "uint16x128")
                c_bits = S.vselr(coeff_bits, coeff_index)
                s_bits = S.vselr(coeff_bits, sin_index)
                c = T.reinterpret(c_bits, "float16x128")
                signed_s = T.reinterpret(S.vxor(s_bits, sign, mask), "float16x128")
                for group in range(TOKEN_ELEMS // FP16_LANES):
                    index = token * TOKEN_ELEMS + group * FP16_LANES
                    values = S.vld(x[index])
                    rotated_bits = S.vselr(T.reinterpret(values, "uint16x128"), partner)
                    rotated = T.reinterpret(rotated_bits, "float16x128")
                    diag = S.vmul(values, c, mask)
                    cross = S.vmul(rotated, signed_s, mask)
                    result = S.vadd(diag, cross, mask)
                    S.vsts(y[index], result, mask, "NORM_B16")

    @T.prim_func
    def main(
        X: T.Tensor((total_k, HEADS, HEAD_DIM), FP16),
        Pos: T.Tensor((total_k,), "int64"),
        Cache: T.Tensor((MAX_SEQ, CACHE_ROW), FP16),
        Y: T.Tensor((total_k, HEADS, HEAD_DIM), FP16),
    ):
        with T.Kernel(blocks) as core:
            src = T.Tensor((total_elems,), FP16, X.data)
            dst = T.Tensor((total_elems,), FP16, Y.data)
            x = T.alloc_shared((tile_elems,), FP16)
            y = T.alloc_shared((tile_elems,), FP16)
            coeff_ub = T.alloc_shared((coeff_elems + coeff_guard,), FP16)
            pos_ub = T.alloc_shared((tile_tokens,), "int64")
            if prefetch:
                T.annotate_buffer_versions({x: 2, y: 2, coeff_ub: 2, pos_ub: 2})
            iterations = groups + T.Cast("int32", core < tail_tiles)
            for step in T.Pipelined(iterations, num_stages=2, annotations=schedule):
                begin = (step * blocks + core) * tile_tokens
                valid = T.min(tile_tokens, total_k - begin)
                base = begin * TOKEN_ELEMS
                if position_prefetch:
                    # Fetch positions one tile ahead of the scalar coefficient lookup.
                    with T.Stage(0):
                        T.copy(Pos[begin : begin + tile_tokens], pos_ub)
                    with T.Stage(1):
                        T.copy(src[base : base + tile_elems], x)
                        load_coefficients(Cache, pos_ub, coeff_ub, valid)
                    with T.Stage(2):
                        rotate(x, y, coeff_ub, valid)
                    with T.Stage(3):
                        T.copy(y, dst[base : base + tile_elems])
                else:
                    T.copy(src[base : base + tile_elems], x)
                    T.copy(Pos[begin : begin + tile_tokens], pos_ub)
                    load_coefficients(Cache, pos_ub, coeff_ub, valid)
                    rotate(x, y, coeff_ub, valid)
                    T.copy(y, dst[base : base + tile_elems])

    return main


def compile_rope(
    x, positions, cache, *, backward=False, num_cores=None, tile_tokens=None, prefetch=False, position_prefetch=False
):
    """Compile FP16 RoPE with each cache row laid out as [cos, sin]."""
    if x.ndim != 3 or x.shape[1:] != (HEADS, HEAD_DIM) or x.dtype != torch.float16:
        raise ValueError("Input must be FP16 with shape (tokens, 56, 32)")
    if positions.shape != (x.shape[0],) or positions.dtype != torch.int64:
        raise ValueError("Expected one int64 position per token")
    if cache.shape != (MAX_SEQ, CACHE_ROW) or cache.dtype != torch.float16:
        raise ValueError("Expected a packed FP16 coefficient table with shape (4096, 64)")
    if any(t.device != x.device for t in (positions, cache)):
        raise ValueError("All RoPE inputs must be on the same device")
    if num_cores is None:
        num_cores = get_num_vector_cores(x.device.index or 0)
    return tilelang.compile(
        fused_rope(
            x.shape[0], num_cores, tile_tokens, backward=backward, prefetch=prefetch, position_prefetch=position_prefetch
        ),
        target="ascend",
    )


def launch(x, positions, cos, sin, *, backward=False, **config):
    """Apply the FP16 RoPE forward or input-gradient kernel."""
    if any(t.shape != (MAX_SEQ, HEAD_DIM) or t.dtype != torch.float16 for t in (cos, sin)):
        raise ValueError("Expected FP16 coefficient tables with shape (4096, 32)")
    cache = torch.cat((cos, sin), dim=-1)
    kernel = compile_rope(x, positions, cache, backward=backward, **config)
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    kernel(x.contiguous(), positions.contiguous(), cache, output)
    return output


def torch_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
) -> torch.Tensor:
    max_seq = cos_cached.shape[0]
    head_dim = x.shape[-1]
    half = head_dim // 2
    positions = torch.clamp(positions, max=max_seq - 1)
    cos = cos_cached[positions].unsqueeze(1)
    sin = sin_cached[positions].unsqueeze(1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rotate_half * sin.to(x.dtype)


def reference(x, positions, cos, sin, backward=False):
    if not backward:
        return torch_ref(x, positions, cos, sin)
    with torch.enable_grad():
        # RoPE is linear in x, so the input gradient is independent of the forward input.
        x_req = x.detach().clone().requires_grad_(True)
        golden_out = torch_ref(x_req, positions, cos, sin)
        (golden_dx,) = torch.autograd.grad(golden_out, x_req, x)
    return golden_dx


def check_accuracy(output, x, positions, cos, sin, backward):
    # Keep peak CPU memory bounded even at 600,000 tokens.
    squared_error = squared_reference = max_error = 0.0
    for begin in range(0, x.shape[0], 4096):
        chunk = slice(begin, begin + 4096)
        expected = reference(x[chunk].cpu(), positions[chunk], cos, sin, backward)
        actual = output[chunk].cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        error = actual.to(torch.float64) - expected.to(torch.float64)
        max_error = max(max_error, error.abs().max().item())
        squared_error += error.square().sum().item()
        squared_reference += expected.to(torch.float64).square().sum().item()
    return max_error, (squared_error / max(squared_reference, 1e-12)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cores", type=int, help="override the device vector core count")
    parser.add_argument("--tile-tokens", type=int)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--prefetch", action=argparse.BooleanOptionalAction, default=False, help="try an offset DMA/SIMD pipeline"
    )
    parser.add_argument(
        "--position-prefetch", action=argparse.BooleanOptionalAction, default=False,
        help="prefetch the next position tile before reading the current one",
    )
    parser.add_argument("--mte2-baseline", action="store_true", help="measure a matched GM-to-UB read baseline per tile")
    parser.add_argument("--dma-baseline", action="store_true", help="measure a matched TileLang DMA copy per tile")
    parser.add_argument("--dma-offset", action="store_true", help="also measure the offset schedule for the TileLang DMA copy")
    args = parser.parse_args()
    if args.iters <= 0 or (args.cores is not None and args.cores <= 0):
        parser.error("iterations and cores must be positive")
    limit = tile_limit()
    if args.tile_tokens is not None and not 1 <= args.tile_tokens <= limit:
        parser.error(f"tile-tokens must be between 1 and {limit}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("An Ascend NPU and torch_npu are required")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    positions = torch.arange(TOTAL_K, dtype=torch.int64) % MAX_SEQ
    positions[-1] = MAX_SEQ  # Exercise the upper clamp.
    freq = 10000.0 ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angles = torch.outer(torch.arange(MAX_SEQ, dtype=torch.float32), freq).repeat(1, 2)
    cos, sin = angles.cos().half(), angles.sin().half()
    # Pack constant tables once and reuse them for every timed launch.
    tables = positions.to(device), torch.cat((cos, sin), dim=-1).to(device)
    shape = (TOTAL_K, HEADS, HEAD_DIM)
    output = torch.empty(shape, device=device, dtype=torch.float16)
    num_cores = args.cores if args.cores is not None else get_num_vector_cores(device.index or 0)
    tokens_per_core = tilelang.cdiv(TOTAL_K, num_cores)
    default_tile = min(tokens_per_core, (limit + 1) // 2)
    if args.tune and args.tile_tokens is None:
        tiles = sorted({min(tokens_per_core, max(1, tile)) for tile in (limit // 4, default_tile, 3 * limit // 4, limit)})
    else:
        tiles = [args.tile_tokens or default_tile]
    bench = partial(
        do_bench, backend="event", device=device, return_mode="median", _n_warmup=10, _n_repeat=args.iters
    )
    io_bytes = 2 * output.numel() * output.element_size()
    read_bytes = io_bytes // 2 + TOTAL_K * (
        positions.element_size() + HEAD_DIM * (cos.element_size() + sin.element_size())
    )
    read_baselines = {}
    dma_copy_times = {}

    for backward in (False, True):
        label = "backward" if backward else "forward"
        x = torch.randn(shape, device=device, dtype=torch.float16)
        if not backward:
            # Reuse the preallocated RoPE buffers for the same-device copy.
            copy_ms = bench(lambda: output.copy_(x))
            if copy_ms <= 0:
                raise RuntimeError("Device timer did not resolve the copy baseline")
            print(f"copy baseline (torch.Tensor.copy_)  {copy_ms:.4f} ms  {io_bytes / (copy_ms * 1e6):.1f} GB/s")
            if args.mte2_baseline:
                for tile_tokens in tiles:
                    read_ms, read_gbps = bench_mte2_read(x, num_cores, tile_tokens, bench)
                    read_baselines[tile_tokens] = read_gbps
                    print(f"GM-to-UB baseline tile={tile_tokens:2}  {read_ms:.4f} ms  {read_gbps:.1f} GB/s")
            if args.dma_baseline or args.dma_offset:
                for tile_tokens in tiles:
                    dma_ms = bench_dma_copy(x, output, num_cores, tile_tokens, bench)
                    dma_copy_times[tile_tokens] = dma_ms
                    print(f"TileLang DMA copy tile={tile_tokens:2}  {dma_ms:.4f} ms  {io_bytes / (dma_ms * 1e6):.1f} GB/s")
                    if args.dma_offset:
                        offset_ms = bench_dma_copy(x, output, num_cores, tile_tokens, bench, offset=True)
                        print(
                            f"TileLang DMA copy offset tile={tile_tokens:2}  {offset_ms:.4f} ms  "
                            f"{io_bytes / (offset_ms * 1e6):.1f} GB/s  speedup={dma_ms / offset_ms:.3f}x"
                        )
        timings = []
        for tile_tokens in tiles:
            kernel = compile_rope(
                x, *tables, num_cores=num_cores, tile_tokens=tile_tokens, backward=backward,
                prefetch=args.prefetch, position_prefetch=args.position_prefetch,
            )
            kernel(x, *tables, output)
            max_error, rel_l2 = check_accuracy(output, x, positions, cos, sin, backward)
            latency = bench(lambda: kernel(x, *tables, output))
            gbps = io_bytes / (latency * 1e6)
            mte2 = ""
            if args.mte2_baseline:
                read_gbps = read_bytes / (latency * 1e6)
                mte2 = f" mte2_read={read_gbps:.1f} GB/s mte2_bw={read_gbps / read_baselines[tile_tokens]:.1%}"
            dma = f" dma_bw={dma_copy_times[tile_tokens] / latency:.1%}" if tile_tokens in dma_copy_times else ""
            timings.append((latency, tile_tokens))
            print(
                f"{label:8} simd tile={tile_tokens:2}  {latency:.4f} ms  {gbps:.1f} GB/s (input/output)  "
                f"copy_bw={copy_ms / latency:.1%}{dma}{mte2}  max_abs={max_error:.3e}  rel_l2={rel_l2:.3e}"
            )
        if args.tune:
            latency, tile_tokens = min(timings)
            print(f"{label:8} simd best tile={tile_tokens:2}  {latency:.4f} ms  copy_bw={copy_ms / latency:.1%}")
        del x
    print("ACCURACY: PASS")


if __name__ == "__main__":
    main()
