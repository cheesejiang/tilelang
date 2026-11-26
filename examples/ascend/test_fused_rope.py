"""FP16 RoPE coverage for a partial token tile and asymmetric caches."""

import pytest
import torch
import tilelang.testing

from example_fused_rope import HEAD_DIM, HEADS, MAX_SEQ, launch, reference


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_fused_rope(backward, prefetch):
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((43, HEADS, HEAD_DIM), generator=generator, dtype=torch.float16)
    positions = torch.randint(0, 2 * MAX_SEQ, (x.shape[0],), generator=generator)
    cos, sin = torch.randn((2, MAX_SEQ, HEAD_DIM), generator=generator, dtype=torch.float16)
    actual = launch(
        x.npu(), positions.npu(), cos.npu(), sin.npu(), num_cores=2, tile_tokens=7, backward=backward, prefetch=prefetch
    ).cpu()
    expected = reference(x, positions, cos, sin, backward)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
