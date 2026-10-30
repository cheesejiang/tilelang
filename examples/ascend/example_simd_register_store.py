"""Show the GM -> UB -> SIMD registers -> UB -> GM data path."""

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def make_add():
    # A 2048-bit vector holds 64 float32 elements (256 bytes).
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            b_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")

            # MTE2 copies inputs into UB. No vector registers yet.
            T.copy(A, a_ub)
            T.copy(B, b_ub)

            with T.SimdVF():
                # 32 is the element bit width, not the number of elements.
                mask = T.simd.pset(32)
                a_reg = T.simd.vld(a_ub[0])  # UB -> vector register
                b_reg = T.simd.vld(b_ub[0])
                sum_reg = T.simd.vadd(a_reg, b_reg, mask)  # register result
                # vadd does not write c_ub. This explicit store does.
                T.simd.vsts(c_ub[0], sum_reg, mask)  # register -> UB

            # MTE3 copies the UB result into GM.
            T.copy(c_ub, C)

    return main


if __name__ == "__main__":
    kernel = tilelang.compile(make_add(), target="ascend", out_idx=-1)
    print(kernel.get_kernel_source())
    a = torch.arange(64, device="npu", dtype=torch.float32)
    b = torch.ones_like(a)
    result = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), a.cpu() + b.cpu())
    print("SIMD register -> UB store passed:", result[:8].cpu())
