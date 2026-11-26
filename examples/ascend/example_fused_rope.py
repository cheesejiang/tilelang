"""Fused rotate-half RoPE forward and input gradient on Ascend.

Run the original workload with `python example_fused_rope.py` or measure
several tile configurations on the current device with `--tune`.
"""

import argparse
from itertools import product

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.carver.arch.driver.ascend_driver import get_num_vector_cores
from tilelang.profiler import do_bench

FLOAT_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
UB_BYTES = 208 * 1024


def tile_limit(heads, dim, dtype, cos_dtype, sin_dtype):
    data_bytes = heads * dim * torch.empty((), dtype=dtype).element_size()
    table_bytes = dim * sum(torch.empty((), dtype=t).element_size() for t in (cos_dtype, sin_dtype))
    # Four data tiles, two coefficient tiles, and space for UB alignment.
    return (UB_BYTES - 6 * 32) // (4 * data_bytes + table_bytes)


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
    threads=512,
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
    limit = tile_limit(heads, dim, dtype, cos_dtype, sin_dtype)
    if tile_tokens is None:
        tile_tokens = min(tilelang.cdiv(total_k, num_cores), max(1, (limit + 1) // 2))
    if not 1 <= tile_tokens <= max(1, limit):
        raise ValueError(f"tile_tokens must be between 1 and {max(1, limit)}")
    dtype, cos_dtype, sin_dtype, position_dtype = (
        str(t).removeprefix("torch.") for t in (dtype, cos_dtype, sin_dtype, position_dtype)
    )
    half = dim // 2
    pairs = heads * half
    blocks = min(num_cores, tilelang.cdiv(total_k, tile_tokens))
    groups, tail_tiles = divmod(tilelang.cdiv(total_k, tile_tokens), blocks)

    @T.macro
    def store_pair(Y, token, head, lane, lo, hi, c_lo, c_hi, s_lo, s_hi):
        if backward:
            # The transpose swaps the sine halves, including for non-NeoX tables.
            Y[token, head, lane] = T.Cast(dtype, lo * c_lo + hi * s_hi)
            Y[token, head, lane + half] = T.Cast(dtype, hi * c_hi - lo * s_lo)
        else:
            Y[token, head, lane] = T.Cast(dtype, lo * c_lo - hi * s_lo)
            Y[token, head, lane + half] = T.Cast(dtype, hi * c_hi + lo * s_hi)

    @T.macro
    def rotate_tile(X, Pos, Cos, Sin, Y, x, y, cos, sin, begin, valid):
        T.copy(X[begin : begin + tile_tokens, :, :], x)
        # Gather each coefficient once per token and share it across all heads.
        with T.SimtVF(threads=128):
            for table_token, table_lane in T.Parallel(tile_tokens, dim, coalesced_width=T.int32(1)):
                if table_token < valid:
                    pos = T.Cast("int32", T.min(Pos[begin + table_token], T.Cast(position_dtype, max_seq - 1)))
                    cos[table_token, table_lane] = Cos[pos, table_lane]
                    sin[table_token, table_lane] = Sin[pos, table_lane]
                else:
                    cos[table_token, table_lane] = T.Cast(cos_dtype, 0)
                    sin[table_token, table_lane] = T.Cast(sin_dtype, 0)
        with T.SimtVF(threads=threads):
            for token, pair in T.Parallel(tile_tokens, pairs, coalesced_width=T.int32(1)):
                head = pair // half
                lane = pair % half
                lo = T.Cast("float32", x[token, head, lane])
                hi = T.Cast("float32", x[token, head, lane + half])
                c_lo = T.Cast("float32", cos[token, lane])
                c_hi = T.Cast("float32", cos[token, lane + half])
                s_lo = T.Cast("float32", sin[token, lane])
                s_hi = T.Cast("float32", sin[token, lane + half])
                store_pair(y, token, head, lane, lo, hi, c_lo, c_hi, s_lo, s_hi)
        T.copy(y, Y[begin : begin + tile_tokens, :, :])

    @T.prim_func
    def main(
        X: T.Tensor((total_k, heads, dim), dtype),
        Pos: T.Tensor((total_k,), position_dtype),
        Cos: T.Tensor((max_seq, dim), cos_dtype),
        Sin: T.Tensor((max_seq, dim), sin_dtype),
        Y: T.Tensor((total_k, heads, dim), dtype),
    ):
        with T.Kernel(blocks) as core:
            if limit == 0:
                # A single very wide token exceeds the double-buffer budget.
                # Stream pairs through registers directly in that case.
                for token in T.serial(core, total_k, blocks):
                    with T.SimtVF(threads=threads):
                        for pair in T.Parallel(pairs, coalesced_width=T.int32(1)):
                            head = pair // half
                            lane = pair % half
                            pos = T.Cast("int32", T.min(Pos[token], T.Cast(position_dtype, max_seq - 1)))
                            lo = T.Cast("float32", X[token, head, lane])
                            hi = T.Cast("float32", X[token, head, lane + half])
                            c_lo = T.Cast("float32", Cos[pos, lane])
                            c_hi = T.Cast("float32", Cos[pos, lane + half])
                            s_lo = T.Cast("float32", Sin[pos, lane])
                            s_hi = T.Cast("float32", Sin[pos, lane + half])
                            store_pair(Y, token, head, lane, lo, hi, c_lo, c_hi, s_lo, s_hi)
            else:
                x = T.alloc_shared((tile_tokens, heads, dim), dtype)
                y = T.alloc_shared((tile_tokens, heads, dim), dtype)
                cos = T.alloc_shared((tile_tokens, dim), cos_dtype)
                sin = T.alloc_shared((tile_tokens, dim), sin_dtype)
                # Give only the first tail_tiles cores an extra iteration.
                iterations = groups + T.Cast("int32", core < tail_tiles)
                if groups > 1:
                    T.annotate_buffer_versions({x: 2, y: 2})
                    for step in T.Pipelined(iterations, num_stages=2):
                        begin = (step * blocks + core) * tile_tokens
                        rotate_tile(X, Pos, Cos, Sin, Y, x, y, cos, sin, begin, T.min(tile_tokens, total_k - begin))
                else:
                    for step in T.serial(iterations):
                        begin = (step * blocks + core) * tile_tokens
                        rotate_tile(X, Pos, Cos, Sin, Y, x, y, cos, sin, begin, T.min(tile_tokens, total_k - begin))

    return main


def compile_rope(x, positions, cos, sin, *, backward=False, num_cores=None, tile_tokens=None, threads=512):
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
            threads=threads,
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
    parser.add_argument("--threads", type=int, default=512)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if min(args.total_k, args.max_seq, args.iters, args.heads, args.head_dim) <= 0 or args.head_dim % 2:
        parser.error("sizes and iterations must be positive, and head-dim must be even")
    if args.cores is not None and args.cores <= 0:
        parser.error("cores must be positive")
    if args.threads not in (128, 256, 512, 1024, 2048):
        parser.error("threads must be 128, 256, 512, 1024 or 2048")
    dtype = FLOAT_DTYPES[args.dtype]
    cache_dtype = FLOAT_DTYPES[args.cache_dtype or args.dtype]
    limit = tile_limit(args.heads, args.head_dim, dtype, cache_dtype, cache_dtype)
    if args.tile_tokens is not None and not 1 <= args.tile_tokens <= max(1, limit):
        parser.error(f"tile-tokens must be between 1 and {max(1, limit)}")
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
    default_tile = min(tokens_per_core, max(1, (limit + 1) // 2))
    tiles = sorted({min(tokens_per_core, max(1, tile)) for tile in (limit // 4, default_tile, 3 * limit // 4, limit)})
    configs = list(product(tiles, (256, 512, 1024))) if args.tune else [(args.tile_tokens or default_tile, args.threads)]

    for backward in (False, True):
        label = "backward" if backward else "forward"
        x = torch.randn(shape, device=device, dtype=dtype)
        timings = []
        for tile_tokens, threads in configs:
            kernel = compile_rope(
                x, *tables, num_cores=num_cores, tile_tokens=tile_tokens, threads=threads, backward=backward
            )
            kernel(x, *tables, output)
            max_error, rel_l2 = check_accuracy(output, x, positions, cos, sin, backward)
            latency = do_bench(
                lambda: kernel(x, *tables, output),
                backend="event",
                device=device,
                return_mode="median",
                _n_warmup=10,
                _n_repeat=args.iters,
            )
            gbps = 2 * x.numel() * x.element_size() / (latency * 1e6)
            timings.append((latency, tile_tokens, threads))
            print(
                f"{label:8} tile={tile_tokens:2} threads={threads:4} "
                f"{latency:.4f} ms  {gbps:.1f} GB/s (input/output)  "
                f"max_abs={max_error:.3e}  rel_l2={rel_l2:.3e}"
            )
        if args.tune:
            latency, tile_tokens, threads = min(timings)
            print(f"{label:8} best tile={tile_tokens} threads={threads}  {latency:.4f} ms")
        del x
    print("ACCURACY: PASS")


if __name__ == "__main__":
    main()
