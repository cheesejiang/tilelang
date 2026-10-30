"""Dependent UB rounds, with an observable intermediate and final output."""

import tilelang.ascend.language as T
from tilelang.layout import Fragment, Layout


def make_reuse_kernel(mode, row_stride=1, blocks=32, columns=32, iterations=64, rotate=False):
    rows = 32 * row_stride
    flat = mode == "ownership"
    shape = (rows * columns,) if flat else (rows, columns)
    pinned = Fragment((32,), forward_thread_fn=lambda i: i, forward_index_fn=lambda i: 0)
    loop_layout = pinned if mode in ("baseline", "padding") else None
    # 64B row pitch needs 16*s; 128B multiples need 8*s.
    period = (16 if (row_stride * columns * 4) % 128 == 64 else 8) * row_stride
    physical = Layout((rows, columns), lambda i, j: i * columns + j + (i // period) * 8)
    shift = 1 if rotate else 0

    @T.prim_func
    def main(A: T.Tensor((blocks * rows, columns), "float32"), C: T.Tensor((blocks * 32,), "float32")):
        with T.Kernel(blocks) as block:
            work = T.alloc_shared(shape, "float32")
            result = T.alloc_shared((32,), "float32")
            if mode == "padding":
                T.annotate_layout({work: physical})
            T.copy(A[block * rows : (block + 1) * rows, :], work)
            with T.SimtVF(threads=32):
                for repeat in T.serial(iterations):
                    for i in T.Parallel(32, loop_layout=loop_layout):
                        if flat:
                            result[i] = work[i * row_stride * columns] * T.float32(0.999) + T.float32(0.001)
                        else:
                            result[i] = work[i * row_stride, 0] * T.float32(0.999) + T.float32(0.001)
                    for i in T.Parallel(32, loop_layout=loop_layout):
                        if flat:
                            work[i * row_stride * columns] = result[(i + shift) % 32]
                        else:
                            work[i * row_stride, 0] = result[(i + shift) % 32]
                for i in T.Parallel(32, loop_layout=loop_layout):
                    if flat:
                        result[i] = work[i * row_stride * columns]
                    else:
                        result[i] = work[i * row_stride, 0]
            T.copy(result, C[block * 32 : (block + 1) * 32])

    return main
