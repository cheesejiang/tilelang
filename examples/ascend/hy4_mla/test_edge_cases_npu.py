"""Device regressions for softmax transitions and the zero-burst workaround."""

import math

import pytest
import torch

from examples.ascend.hy4_mla.compile_sparse_fwd import compile_bf16
from examples.ascend.hy4_mla.testing import check_outputs, sparse_mla_reference


@pytest.fixture
def npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("sink_logit", [80.0, 90.0, 100.0])
def test_large_finite_sink_preserves_representable_output(npu, pipeline, sink_logit):
    # Large values amplify the small attention mass into a normal BF16 result.
    # Both the expected output and the unnormalized PV sum remain finite.
    q = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
    kv = torch.zeros((1, 576), dtype=torch.bfloat16)
    kv[:, :512] = 1e36
    indices = torch.zeros((1, 64), dtype=torch.int32)
    attn_sink = torch.full((64,), sink_logit, dtype=torch.float32)
    lengths = torch.full((1,), 64, dtype=torch.int32)
    # Evaluate with a common maximum; exp(sink_logit) need never be formed.
    weight = 64 * math.exp(-sink_logit) / (1 + 64 * math.exp(-sink_logit))
    expected = torch.full((1, 64, 512), float(kv[0, 0]) * weight, dtype=torch.bfloat16)
    assert (expected > 0).all()
    kernel = compile_bf16(1, 1, 64, scale=1.0, pipeline_stages=pipeline, num_cores=1)
    actual = kernel(*(t.to(npu) for t in (q, kv, indices, attn_sink, lengths)))
    torch.npu.synchronize()
    # Absolute tolerance would conceal losing the entire small output.
    torch.testing.assert_close(actual[0].cpu(), expected, rtol=1e-2, atol=0)
    torch.testing.assert_close(actual[1].cpu(), torch.zeros((1, 64)), rtol=0, atol=0)
    torch.testing.assert_close(actual[2].cpu(), torch.full((1, 64), math.log(64)), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("topk", [128, 192, 704, 2048])
def test_multi_tile_rescale_with_persistent_jobs(npu, pipeline, topk):
    nq, nk = 5, topk
    q = torch.zeros((nq, 64, 576), dtype=torch.bfloat16)
    q[:, :, 512:] = 1
    kv = torch.zeros((nk, 576), dtype=torch.bfloat16)
    indices = torch.arange(topk, dtype=torch.int32).repeat(nq, 1)
    attn_sink = torch.zeros(64, dtype=torch.float32)
    lengths = torch.full((nq,), topk, dtype=torch.int32)
    kernel = compile_bf16(nq, nk, topk, scale=0.5, sink=False, pipeline_stages=pipeline, num_cores=2)
    for sign in (1, -1, 1):
        for tile in range(topk // 64):
            kv[tile * 64 : (tile + 1) * 64, :512] = sign * (tile + 1)
            # Force a rescale in the second and final tile, including when
            # there are intervening tiles that accumulate without rescaling.
            kv[tile * 64 : (tile + 1) * 64, 512:] = (tile > 0) + (tile == topk // 64 - 1)
        inputs = q, kv, indices, attn_sink, lengths
        expected = sparse_mla_reference(*inputs, scale=0.5, sink=False)
        actual = kernel(*(t.to(npu) for t in inputs))
        torch.npu.synchronize()
        check_outputs(actual, expected)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("topk,first_valid_tile", [(320, 2), (704, 10), (2048, 10)])
def test_invalid_prefix_then_valid_tile(npu, pipeline, topk, first_valid_tile):
    generator = torch.Generator().manual_seed(43)
    nq, nk = 5, 79
    q = (torch.randn((nq, 64, 576), generator=generator) * 0.3).to(torch.bfloat16)
    kv = (torch.randn((nk, 576), generator=generator) * 0.3).to(torch.bfloat16)
    attn_sink = torch.zeros(64, dtype=torch.float32)
    lengths = torch.full((nq,), topk, dtype=torch.int32)
    kernel = compile_bf16(nq, nk, topk, scale=1 / 16, variable_lengths=True, pipeline_stages=pipeline, num_cores=2)
    for valid_tile in (first_valid_tile, 0, first_valid_tile):
        indices = torch.full((nq, topk), -1, dtype=torch.int32)
        indices[:, 1::2] = nk
        # Row zero stays empty. Other rows contain a full valid pair plus
        # both orders of mixed valid/invalid pairs, surrounded by empty tiles.
        start = valid_tile * 64
        indices[1:, start : start + 4] = torch.tensor([3, 7, -1, 11], dtype=torch.int32)
        indices[1:, start + 4 : start + 6] = torch.tensor([13, nk], dtype=torch.int32)
        inputs = q, kv, indices, attn_sink, lengths
        expected = sparse_mla_reference(*inputs, scale=1 / 16)
        actual = kernel(*(t.to(npu) for t in inputs))
        torch.npu.synchronize()
        check_outputs(actual, expected)


def test_zero_burst_copy_preserves_ub(npu):
    import tilelang
    import tilelang.ascend.language as T

    @T.prim_func
    def copy(
        source: T.Tensor((2, 576), "bfloat16"),
        initial: T.Tensor((2, 640), "bfloat16"),
        output: T.Tensor((2, 640), "bfloat16"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((2, 640), "bfloat16")
            T.copy(initial, ub)
            n_burst = T.alloc_var("int32", init=T.if_then_else(enabled != 0, 2, 0))
            # Match the kernel's workaround with a runtime count. Distinct
            # source and initial values expose an unwanted nonzero transfer.
            for _ in T.Serial(1):
                T.assume(n_burst > 0)
                T.copy(source[:n_burst, :], ub[:n_burst, :576])
            T.copy(ub, output)

    kernel = tilelang.compile(copy, target="ascend")
    initial = torch.full((2, 640), -7, dtype=torch.bfloat16, device=npu)
    output = torch.empty_like(initial)
    for enabled in (0, 1, 0, 1, 0):
        source = torch.full((2, 576), 3 + enabled, dtype=torch.bfloat16, device=npu)
        kernel(source, initial, output, enabled)
        torch.npu.synchronize()
        expected = initial.clone()
        if enabled:
            expected[:, :576] = source
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # This checks destination preservation, not the absence of a discarded GM
    # read; proving no read also requires generated-code or device tracing.
