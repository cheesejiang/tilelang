"""SIMT stage extracts, preserving arithmetic rather than entire source kernels."""

import tilelang.ascend.language as T


def make_kernel(case, rows=32, columns=32, threads=32, blocks=32, updates=4):
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
