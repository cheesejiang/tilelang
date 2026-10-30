"""SIMT stage extracts, preserving arithmetic rather than entire source kernels."""

import tilelang.ascend.language as T
from tilelang.layout import Fragment, Layout


def make_kernel(case, mode, rows=32, columns=32, threads=32, blocks=32, updates=4):
    cross = case not in ("vecadd", "gemm_bias")
    flat = not cross or mode == "ownership"
    n = rows * columns
    shape = (n,) if flat else (rows, columns)
    # Cross-row loops enumerate (column, row), so the last logical axis is
    # the row axis. Auto receives no explicit loop or storage annotations.
    cross_partition = Fragment(
        (columns, rows), forward_thread_fn=lambda c, r: r % threads,
        forward_index_fn=lambda c, r: c * (rows // threads) + r // threads,
    )
    contiguous_partition = Fragment(
        (n,), forward_thread_fn=lambda i: (i // 2) % threads,
        forward_index_fn=lambda i: (i // (2 * threads)) * 2 + i % 2,
    )
    loop_layout = (cross_partition if cross else contiguous_partition) if mode in ("baseline", "padding") else None
    physical = Layout((rows, columns), lambda r, c: r * columns + c + (r // 8) * 8)

    @T.prim_func
    def main(
        A: T.Tensor((blocks, rows, columns), "float32"),
        B: T.Tensor((blocks, updates, rows, columns), "float32"),
        Stats: T.Tensor((blocks, updates + 1, rows), "float32"),
        C: T.Tensor((blocks, rows, columns), "float32"),
    ):
        with T.Kernel(blocks) as block:
            work = T.alloc_shared(shape, "float32")
            if cross and mode == "padding":
                T.annotate_layout({work: physical})
            T.copy(A[block, :, :], work)
            if case == "vecadd":
                incoming = T.alloc_shared((n,), "float32")
                T.copy(B[block, 0, :, :], incoming)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(n, loop_layout=loop_layout):
                        work[i] = work[i] + incoming[i]
            elif case == "gemm_bias":
                bias = T.alloc_shared((columns,), "float32")
                T.copy(B[block, 0, 0, :], bias)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(n, loop_layout=loop_layout):
                        work[i] = work[i] + bias[i % columns]
            elif case == "gemm_scale":
                with T.SimtVF(threads=threads):
                    for c, r in T.Parallel(columns, rows, loop_layout=loop_layout):
                        if flat:
                            work[r * columns + c] = work[r * columns + c] * T.float32(1.25)
                        else:
                            work[r, c] = work[r, c] * T.float32(1.25)
            elif case == "attention_finalize":
                denominator = T.alloc_shared((rows,), "float32")
                T.copy(Stats[block, updates, :], denominator)
                with T.SimtVF(threads=threads):
                    for c, r in T.Parallel(columns, rows, loop_layout=loop_layout):
                        if flat:
                            work[r * columns + c] = work[r * columns + c] / denominator[r]
                        else:
                            work[r, c] = work[r, c] / denominator[r]
            else:
                alpha = T.alloc_shared((rows,), "float32")
                incoming = T.alloc_shared(shape, "float32")
                if mode == "padding":
                    T.annotate_layout({incoming: physical})
                # Each iteration consumes a DIFFERENT GM tile, as in the
                # online attention output accumulator; not repeated dummy math.
                for kv in T.serial(updates):
                    T.copy(Stats[block, kv, :], alpha)
                    T.copy(B[block, kv, :, :], incoming)
                    with T.SimtVF(threads=threads):
                        for c, r in T.Parallel(columns, rows, loop_layout=loop_layout):
                            if flat:
                                work[r * columns + c] = work[r * columns + c] * alpha[r] + incoming[r * columns + c]
                            else:
                                work[r, c] = work[r, c] * alpha[r] + incoming[r, c]
                if case == "attention_update_finalize":
                    denominator = T.alloc_shared((rows,), "float32")
                    T.copy(Stats[block, updates, :], denominator)
                    with T.SimtVF(threads=threads):
                        for c, r in T.Parallel(columns, rows, loop_layout=loop_layout):
                            if flat:
                                work[r * columns + c] = work[r * columns + c] / denominator[r]
                            else:
                                work[r, c] = work[r, c] / denominator[r]
            T.copy(work, C[block, :, :])

    return main
