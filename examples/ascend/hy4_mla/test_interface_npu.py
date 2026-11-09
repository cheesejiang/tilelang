"""Device acceptance tests for the BF16 call adapter; not run by CPU tests."""

import pytest
import torch

from examples.ascend.hy4_mla.interface import flash_mla_sparse_fwd
from examples.ascend.hy4_mla.testing import sparse_mla_reference


@pytest.fixture
def npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize(
    "heads,use_sink,use_lengths,pitched",
    [(64, False, False, False), (64, True, True, True), (8, True, False, False), (16, False, True, False), (32, True, True, False)],
)
def test_bf16_adapter(npu, heads, use_sink, use_lengths, pitched):
    generator = torch.Generator().manual_seed(19)
    nq, nk, topk = 3, 79, 64
    q_cpu = (torch.randn((nq, heads, 576), generator=generator) * 0.3).to(torch.bfloat16)
    kv_cpu = (torch.randn((nk, 1, 576), generator=generator) * 0.3).to(torch.bfloat16)
    indices_cpu = torch.randint(nk, (nq, 1, topk), generator=generator, dtype=torch.int32)
    indices_cpu[:, :, ::11] = -1
    indices_cpu[:, :, 3::17] = nk
    sink_cpu = torch.linspace(-3, 3, heads) if use_sink else None
    if use_sink:
        sink_cpu[0], sink_cpu[-1] = -torch.inf, torch.inf
    lengths_cpu = torch.tensor([64, 0, 1], dtype=torch.int32) if use_lengths else None

    reference_q = torch.zeros((nq, 64, 576), dtype=torch.bfloat16)
    reference_q[:, :heads].copy_(q_cpu)
    reference_sink = torch.full((64,), -torch.inf)
    if use_sink:
        reference_sink[:heads].copy_(sink_cpu)
    expected = sparse_mla_reference(
        reference_q, kv_cpu.squeeze(1), indices_cpu.squeeze(1), reference_sink, lengths_cpu, scale=1 / 16, sink=use_sink
    )
    q, kv, indices = (tensor.to(npu) for tensor in (q_cpu, kv_cpu, indices_cpu))
    if pitched:
        pitched_q = torch.empty((nq, 128, 600), dtype=q.dtype, device=npu)[:, :heads, :576]
        pitched_kv = torch.empty((nk, 1, 600), dtype=kv.dtype, device=npu)[:, :, :576]
        pitched_indices = torch.empty((nq, 1, 80), dtype=indices.dtype, device=npu)[:, :, :topk]
        pitched_q.copy_(q)
        pitched_kv.copy_(kv)
        pitched_indices.copy_(indices)
        q, kv, indices = pitched_q, pitched_kv, pitched_indices
        out_backing = torch.full((nq, 128, 520), -9, dtype=q.dtype, device=npu)
        out = out_backing[:, :heads, :512]
    else:
        out = torch.empty((nq, heads, 512), dtype=q.dtype, device=npu)
    sink = sink_cpu.to(npu) if use_sink else None
    lengths = lengths_cpu.to(npu) if use_lengths else None
    for _ in range(2):
        actual = flash_mla_sparse_fwd(q, kv, indices, 1 / 16, attn_sink=sink, topk_length=lengths, out=out)
        torch.npu.synchronize()
        assert actual[0] is out
        for i, (value, ref) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(value.cpu(), ref[:, :heads], rtol=1e-2 if i == 0 else 2e-4, atol=1e-2 if i == 0 else 2e-4)
        if pitched:
            assert (out_backing[:, heads:] == -9).all().item()
            assert (out_backing[:, :heads, 512:] == -9).all().item()
