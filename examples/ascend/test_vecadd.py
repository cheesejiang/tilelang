"""SIMT and SIMD vector addition with reuse of pipeline buffers."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_vecadd import vector_add


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("mode", ["simt", "simd"])
def test_vecadd(mode):
    n = 2**21  # Four iterations per core reuse both buffer versions.
    a, b = torch.randn(n), torch.randn(n)
    kernel = tilelang.compile(vector_add(n, mode), out_idx=-1)
    torch.testing.assert_close(kernel(a.npu(), b.npu()).cpu(), a + b)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("mode", ["simt", "simd"])
@pytest.mark.parametrize("num_tiles,blocks", [(3, 8), (17, 8), (16, 8)])
def test_vecadd_block_partition(mode, num_tiles, blocks):
    n = num_tiles * 8192
    a, b = torch.randn(n, device="npu"), torch.randn(n, device="npu")
    out = torch.full_like(a, float("nan"))
    kernel = tilelang.compile(vector_add(n, mode, num_blocks=blocks), target="ascend")
    kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)
