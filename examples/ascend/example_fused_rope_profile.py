"""Single-launch BF16 RoPE with position prefetch for msprof profiling."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.carver.arch.driver.ascend_driver import get_num_vector_cores
from tilelang.profiler import do_bench

TOTAL_K = 600_000
HEADS = 56
HEAD_DIM = 32
MAX_SEQ = 4096
BF16 = "bfloat16"
BF16_BYTES = 2
BF16_LANES = 128
UB_BYTES = 248 * 1024
TOKEN_ELEMS = HEADS * HEAD_DIM
CACHE_ROW = 2 * HEAD_DIM
S = T.simd


def tile_limit():
    # Two versions of input, output, packed coefficients, and positions.
    coeff_guard = BF16_LANES - CACHE_ROW
    reserve = 8 * 32 + 2 * coeff_guard * BF16_BYTES
    bytes_per_token = 4 * (TOKEN_ELEMS + HEAD_DIM) * BF16_BYTES + 2 * 8
    return (UB_BYTES - reserve) // bytes_per_token


def tile_grid(total_k, num_cores, tile_tokens):
    num_tiles = tilelang.cdiv(total_k, tile_tokens)
    blocks = min(num_cores, num_tiles)
    groups, tail_tiles = divmod(num_tiles, blocks)
    return blocks, groups, tail_tiles


def fused_rope(total_k, num_cores, tile_tokens=16, backward=False, prefetch=True, position_prefetch=True):
    """Build the 56-head, 32-element BF16 kernel."""
    limit = tile_limit()
    if min(total_k, num_cores) <= 0:
        raise ValueError("Expected positive token and core counts")
    if not 1 <= tile_tokens <= limit:
        raise ValueError(f"tile_tokens must be between 1 and {limit}")
    blocks, groups, tail_tiles = tile_grid(total_k, num_cores, tile_tokens)
    total_elems = total_k * TOKEN_ELEMS
    tile_elems = tile_tokens * TOKEN_ELEMS
    coeff_elems = tile_tokens * CACHE_ROW
    coeff_guard = BF16_LANES - CACHE_ROW
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
            # Flip the BF16 sign bit without widening the coefficient.
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
                c = T.reinterpret(c_bits, "bfloat16x128")
                signed_s = T.reinterpret(S.vxor(s_bits, sign, mask), "bfloat16x128")
                for group in range(TOKEN_ELEMS // BF16_LANES):
                    index = token * TOKEN_ELEMS + group * BF16_LANES
                    values = S.vld(x[index])
                    rotated_bits = S.vselr(T.reinterpret(values, "uint16x128"), partner)
                    rotated = T.reinterpret(rotated_bits, "bfloat16x128")
                    diag = S.vmul(values, c, mask)
                    cross = S.vmul(rotated, signed_s, mask)
                    result = S.vadd(diag, cross, mask)
                    S.vsts(y[index], result, mask, "NORM_B16")

    @T.prim_func
    def main(
        X: T.Tensor((total_k, HEADS, HEAD_DIM), BF16),
        Pos: T.Tensor((total_k,), "int64"),
        Cache: T.Tensor((MAX_SEQ, CACHE_ROW), BF16),
        Y: T.Tensor((total_k, HEADS, HEAD_DIM), BF16),
    ):
        with T.Kernel(blocks) as core:
            src = T.Tensor((total_elems,), BF16, X.data)
            dst = T.Tensor((total_elems,), BF16, Y.data)
            x = T.alloc_shared((tile_elems,), BF16)
            y = T.alloc_shared((tile_elems,), BF16)
            coeff_ub = T.alloc_shared((coeff_elems + coeff_guard,), BF16)
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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cores", type=int, help="override the selected device's AIV core count")
    parser.add_argument("--tile-tokens", type=int, default=16)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--bench", action="store_true", help="measure device latency after the first launch")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    if args.cores is not None and args.cores <= 0:
        parser.error("cores must be positive")
    if not 1 <= args.tile_tokens <= tile_limit():
        parser.error(f"tile-tokens must be between 1 and {tile_limit()}")
    if args.iters <= 0:
        parser.error("iters must be positive")

    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    device_index = device.index if device.index is not None else torch.npu.current_device()
    num_cores = args.cores if args.cores is not None else get_num_vector_cores(device_index)
    print(f"RoPE device={device} AIV cores={num_cores} tile={args.tile_tokens}", flush=True)
    generator = torch.Generator().manual_seed(7)
    # Initialize on CPU so profiling sees no NPU initialization kernels.
    x = torch.randn((TOTAL_K, HEADS, HEAD_DIM), generator=generator, dtype=torch.bfloat16).to(args.device)
    positions = torch.arange(TOTAL_K, dtype=torch.int64) % MAX_SEQ
    positions[-1] = MAX_SEQ
    positions = positions.to(args.device)
    frequencies = 10000.0 ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angles = torch.outer(torch.arange(MAX_SEQ, dtype=torch.float32), frequencies).repeat(1, 2)
    cache = torch.cat((angles.cos().bfloat16(), angles.sin().bfloat16()), dim=-1).to(args.device)
    output = torch.empty_like(x)
    torch.npu.synchronize()

    kernel = tilelang.compile(
        fused_rope(TOTAL_K, num_cores, args.tile_tokens, backward=args.backward),
        target="ascend",
    )
    kernel(x, positions, cache, output)
    torch.npu.synchronize()
    if args.bench:
        latency = do_bench(
            lambda: kernel(x, positions, cache, output),
            backend="event",
            device=device,
            return_mode="median",
            _n_warmup=10,
            _n_repeat=args.iters,
        )
        bandwidth = 2 * x.numel() * x.element_size() / (latency * 1e6)
        direction = "backward" if args.backward else "forward"
        print(f"{direction} BF16 tile={args.tile_tokens}  {latency * 1000:.2f} us  {bandwidth:.1f} GB/s")
    else:
        print("RoPE launch completed")


if __name__ == "__main__":
    main()
