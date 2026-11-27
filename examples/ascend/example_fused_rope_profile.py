"""Single-launch FP16 RoPE with position prefetch for msprof profiling."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T

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


def fused_rope(total_k, num_cores, tile_tokens=None, backward=False, prefetch=True, position_prefetch=True):
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
    def fused_rope_fp16(
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

    return fused_rope_fp16



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cores", type=int, default=28)
    parser.add_argument("--tile-tokens", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if args.cores <= 0:
        parser.error("cores must be positive")
    if not 1 <= args.tile_tokens <= tile_limit():
        parser.error(f"tile-tokens must be between 1 and {tile_limit()}")

    torch.npu.set_device(args.device)
    generator = torch.Generator().manual_seed(7)
    # Initialize on CPU so profiling sees no NPU initialization kernels.
    x = torch.randn((TOTAL_K, HEADS, HEAD_DIM), generator=generator, dtype=torch.float16).to(args.device)
    positions = torch.arange(TOTAL_K, dtype=torch.int64) % MAX_SEQ
    positions[-1] = MAX_SEQ
    positions = positions.to(args.device)
    frequencies = 10000.0 ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    angles = torch.outer(torch.arange(MAX_SEQ, dtype=torch.float32), frequencies).repeat(1, 2)
    cache = torch.cat((angles.cos().half(), angles.sin().half()), dim=-1).to(args.device)
    output = torch.empty_like(x)
    torch.npu.synchronize()

    kernel = tilelang.compile(
        fused_rope(TOTAL_K, args.cores, args.tile_tokens, backward=args.backward),
        target="ascend",
    )
    kernel(x, positions, cache, output)
    torch.npu.synchronize()
    print("RoPE launch completed")


if __name__ == "__main__":
    main()
