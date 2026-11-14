"""Check logical-block coverage below, at, and above physical core counts.

Run on an A5 board with the Ascend backend from this checkout. No inter-block
barriers are used: this checks dispatch coverage, not simultaneous residency.

Both decorated JIT factories and explicit PrimFunc construction/compilation
are exercised, using identical kernel bodies and output checks.
"""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T


def _vector_kernel(blocks):
    @T.prim_func
    def main(out: T.Tensor((blocks, 32), "float32")):
        with T.Kernel(blocks) as bx:
            buf = T.alloc_shared((32,), "float32")
            with T.SimtVF(threads=32):
                for i in T.Parallel(32):
                    buf[i] = T.cast(bx + 1, "float32")
            T.copy(buf, out[bx, :])

    return main


def _matrix_kernel(blocks, mixed):
    @T.prim_func
    def main(
        a: T.Tensor((blocks, 16, 16), "float16"),
        b: T.Tensor((16, 16), "float16"),
        out: T.Tensor((blocks, 16, 16), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            al1 = T.alloc_l1((16, 16), "float16")
            bl1 = T.alloc_l1((16, 16), "float16")
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a[bx, :, :], al1)
            T.copy(b, bl1)
            T.gemm(al1, bl1, acc, transpose_B=True, clear_accum=True)
            if mixed:
                ub = T.alloc_shared((8, 16), "float32")
                T.dual_copy(acc, ub)
                T.dual_copy(ub, out[bx, :, :])
            else:
                T.copy(acc, out[bx, :, :])

    return main


@tilelang.jit(target="ascend", execution_backend="tvm_ffi", mode="lazy")
def _jit_vector_kernel(blocks):
    return _vector_kernel(blocks)


@tilelang.jit(target="ascend", execution_backend="tvm_ffi", mode="lazy")
def _jit_matrix_kernel(blocks, mixed):
    return _matrix_kernel(blocks, mixed)


def _build_kernel(mode, blocks, build_path):
    if build_path == "jit":
        if mode == "vector":
            return _jit_vector_kernel(blocks)
        return _jit_matrix_kernel(blocks, mode == "mixed")

    # Same pattern as layout_workloads: construct a PrimFunc, explicitly
    # compile it, then invoke the returned callable separately.
    func = _vector_kernel(blocks) if mode == "vector" else _matrix_kernel(blocks, mode == "mixed")
    return tilelang.compile(func, target="ascend", execution_backend="tvm_ffi")


@pytest.mark.parametrize("build_path", ["jit", "explicit_compile"])
@pytest.mark.parametrize("mode", ["vector", "cube", "mixed"])
@pytest.mark.parametrize("grid_case", ["below", "equal", "above", "double", "fixed"])
def test_block_grid_oversubscription(mode, grid_case, build_path):
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("An Ascend NPU is required")
    device = torch.npu.current_device()
    props = torch.npu.get_device_properties(device)
    cores = int(props.vector_core_num if mode == "vector" else props.cube_core_num)
    assert cores > 0
    blocks = {
        "below": max(1, cores - 1),
        "equal": cores,
        "above": cores + 1,
        "double": 2 * cores,
        "fixed": 64 if mode == "vector" else 32,
    }[grid_case]
    print(f"device={props.name} path={build_path} mode={mode} physical_cores={cores} blocks={blocks}")

    # Exact integer-valued results encode the logical block ID. Sentinel-filled
    # explicit outputs make missing blocks observable, including on repeat runs.
    ids = torch.arange(1, blocks + 1, dtype=torch.float32, device=f"npu:{device}")
    kernel = _build_kernel(mode, blocks, build_path)
    if mode == "vector":
        expected = ids[:, None].expand(blocks, 32)
        inputs = ()
    else:
        a = ids[:, None, None].expand(blocks, 16, 16).to(torch.float16).contiguous()
        b = torch.eye(16, dtype=torch.float16, device=f"npu:{device}")
        expected = ids[:, None, None].expand(blocks, 16, 16)
        inputs = (a, b)
    out = torch.empty(expected.shape, dtype=torch.float32, device=f"npu:{device}")
    for _ in range(3):
        out.fill_(-1)
        kernel(*inputs, out)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)
