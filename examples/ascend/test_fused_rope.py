"""RoPE coverage for mixed dtypes, partial tiles and asymmetric caches."""

import pytest
import torch
import tilelang.testing

from example_fused_rope import launch, reference


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize(
    "mode,shape,dtype,cos_dtype,sin_dtype,position_dtype",
    [
        ("simt", (43, 56, 32), torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int64),
        ("simt", (2, 21, 3, 30), torch.float16, torch.float32, torch.float16, torch.int32),
        ("simt", (43, 5, 64), torch.float32, torch.float16, torch.bfloat16, torch.int64),
        ("simt", (3, 224, 128), torch.float32, torch.float32, torch.float32, torch.int32),
        ("simd", (43, 56, 32), torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int64),
        ("simd", (43, 56, 30), torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int64),
        ("simd", (2, 21, 3, 30), torch.float16, torch.float32, torch.float16, torch.int32),
        ("simd", (43, 5, 160), torch.float32, torch.float16, torch.bfloat16, torch.int32),
    ],
)
def test_fused_rope(backward, mode, shape, dtype, cos_dtype, sin_dtype, position_dtype):
    max_seq = 17
    heads, dim = shape[-2:]
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((*shape[:-2], dim, heads), generator=generator).transpose(-1, -2).to(dtype)
    positions = torch.randint(0, 2 * max_seq, shape[:-2], generator=generator, dtype=position_dtype)
    angles = torch.randn((max_seq, dim), generator=generator)
    cos, sin = angles.cos().to(cos_dtype), angles.sin().to(sin_dtype)
    tile_tokens = 8 if positions.numel() > 8 else None
    actual = launch(
        x.npu(), positions.npu(), cos.npu(), sin.npu(), num_cores=2, tile_tokens=tile_tokens, backward=backward, mode=mode,
    ).cpu()
    expected = reference(x.reshape(-1, heads, dim), positions.flatten(), cos, sin, backward).reshape(shape)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
