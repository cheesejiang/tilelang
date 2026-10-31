"""SIMT stage extracts, preserving arithmetic rather than entire source kernels."""

import tilelang.ascend.language as T

if __package__:
    from .catalog import FLASH_KV_WIDTH
else:
    from catalog import FLASH_KV_WIDTH


def make_kernel(case, rows=32, columns=32, threads=32, blocks=32, updates=4):
    if case == "flash_decode_column_store":
        return _make_flash_decode_column_kernel(rows, columns, threads, blocks)
    if case in ("kda_column_store", "kda_reduce_column_store"):
        return _make_kda_column_kernel(case, rows, columns, threads, blocks, updates)
    # Mode is deliberately absent from the kernel factory: all four runs
    # consume the same unannotated IR; only PassContext options differ.
    cross = case not in ("vecadd", "gemm_bias")
    n = rows * columns
    shape = (rows, columns) if cross else (n,)

    @T.prim_func
    def main(
        A: T.Tensor((blocks, rows, columns), "float32"),
        B: T.Tensor((blocks, updates, rows, columns), "float32"),
        Stats: T.Tensor((blocks, updates + 1, rows), "float32"),
        C: T.Tensor((blocks, rows, columns), "float32"),
    ):
        with T.Kernel(blocks) as block:
            work = T.alloc_shared(shape, "float32")
            T.copy(A[block, :, :], work)
            if case == "vecadd":
                incoming = T.alloc_shared((n,), "float32")
                T.copy(B[block, 0, :, :], incoming)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(n):
                        work[i] = work[i] + incoming[i]
            elif case == "gemm_bias":
                bias = T.alloc_shared((columns,), "float32")
                T.copy(B[block, 0, 0, :], bias)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(n):
                        work[i] = work[i] + bias[i % columns]
            elif case == "gemm_scale":
                with T.SimtVF(threads=threads):
                    for c, r in T.Parallel(columns, rows):
                        work[r, c] = work[r, c] * T.float32(1.25)
            elif case == "attention_finalize":
                denominator = T.alloc_shared((rows,), "float32")
                T.copy(Stats[block, updates, :], denominator)
                with T.SimtVF(threads=threads):
                    for c, r in T.Parallel(columns, rows):
                        work[r, c] = work[r, c] / denominator[r]
            else:
                alpha = T.alloc_shared((rows,), "float32")
                incoming = T.alloc_shared(shape, "float32")
                # Each iteration consumes a DIFFERENT GM tile, as in the
                # online attention output accumulator; not repeated dummy math.
                for kv in T.serial(updates):
                    T.copy(Stats[block, kv, :], alpha)
                    T.copy(B[block, kv, :, :], incoming)
                    with T.SimtVF(threads=threads):
                        for c, r in T.Parallel(columns, rows):
                            work[r, c] = work[r, c] * alpha[r] + incoming[r, c]
                if case == "attention_update_finalize":
                    denominator = T.alloc_shared((rows,), "float32")
                    T.copy(Stats[block, updates, :], denominator)
                    with T.SimtVF(threads=threads):
                        for c, r in T.Parallel(columns, rows):
                            work[r, c] = work[r, c] / denominator[r]
            T.copy(work, C[block, :, :])

    return main


def _make_flash_decode_column_kernel(rows, columns, threads, blocks):
    """Independent per-tile row maxima -> one output column per iteration.

    A/Stats keep the common runner signature but are unused. Input logits are
    supplied in B; this is the reduce/writeback stage, not a full attention.
    All modes use the native reducer result placement; no pinned fragments.
    """
    width = FLASH_KV_WIDTH

    @T.prim_func
    def main(
        A: T.Tensor((blocks, rows, columns), "float32"),
        B: T.Tensor((blocks, columns, rows, width), "float32"),
        Stats: T.Tensor((blocks, columns + 1, rows), "float32"),
        C: T.Tensor((blocks, rows, columns), "float32"),
    ):
        with T.Kernel(blocks) as block:
            logits = T.alloc_shared((rows, width), "float32")
            scores_out = T.alloc_shared((rows, columns), "float32")
            for k in T.serial(columns):
                T.copy(B[block, k, :, :], logits)
                with T.SimtVF(threads=threads):
                    maxima = T.alloc_reducer((rows,), "float32", op="max")
                    T.reducer_init(maxima)
                    for h, n in T.Parallel(rows, width):
                        T.reducer_update(maxima[h], logits[h, n])
                    result = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(maxima, result)
                    T.copy(result, scores_out[:, k])
            T.copy(scores_out, C[block, :, :])

    return main


def _make_kda_column_kernel(case, rows, columns, threads, blocks, updates):
    """One KDA column-store epoch; all four modes share this exact input IR.

    A is both the current output matrix and, for the full stage, one dot
    operand. B[:,0] is the other operand. Non-target output columns retain A.
    Stats[:,0] contains precomputed dot products in the isolated store case.
    This models the j%CS == 0 epoch, not the whole recurrent KDA kernel.
    """

    @T.prim_func
    def main(
        A: T.Tensor((blocks, rows, columns), "float32"),
        B: T.Tensor((blocks, updates, rows, columns), "float32"),
        Stats: T.Tensor((blocks, updates + 1, rows), "float32"),
        C: T.Tensor((blocks, rows, columns), "float32"),
    ):
        with T.Kernel(blocks) as block:
            work = T.alloc_shared((rows, columns), "float32")
            T.copy(A[block, :, :], work)
            if case == "kda_column_store":
                reduced = T.alloc_shared((rows,), "float32")
                T.copy(Stats[block, 0, :], reduced)
                with T.SimtVF(threads=threads):
                    for r in T.Parallel(rows):
                        work[r, 0] = reduced[r]
            else:
                incoming = T.alloc_shared((rows, columns), "float32")
                T.copy(B[block, 0, :, :], incoming)
                with T.SimtVF(threads=threads):
                    sums = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(sums)
                    for r, d in T.Parallel(rows, columns):
                        T.reducer_update(sums[r], work[r, d] * incoming[r, d])
                    reduced = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sums, reduced)
                    for r in T.Parallel(rows):
                        work[r, 0] = reduced[r]
            T.copy(work, C[block, :, :])

    return main
