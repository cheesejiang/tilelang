"""CPU tensor tests of the call adapter with a mock executor, not NPU tests."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from examples.ascend.hy4_mla import interface
from examples.ascend.hy4_mla.compile_sparse_fwd import compile_bf16
from examples.ascend.hy4_mla.testing import sparse_mla_reference


class CallAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.launches = []
        self.compiles = []

        def mock_compile(nq, nk, topk, scale, sink, variable_lengths, q_strides, kv_stride, index_stride, out_strides, device):
            self.compiles.append((nq, nk, topk, scale, sink, variable_lengths, q_strides, kv_stride, index_stride, out_strides, device))

            def run(q, kv, indices, attn_sink, lengths, output, maximum, lse):
                self.launches.append((q, kv, indices, attn_sink, lengths, output, maximum, lse))
                expected = sparse_mla_reference(q, kv, indices, attn_sink, lengths if variable_lengths else None, scale=scale, sink=sink)
                for dest, src in zip((output, maximum, lse), expected):
                    dest.copy_(src)

            return run

        self.compiler = Mock(side_effect=mock_compile)
        self.patcher = patch.object(interface, "_get_kernel", self.compiler)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @staticmethod
    def inputs(heads=64, nq=3):
        generator = torch.Generator().manual_seed(17)
        q = torch.randn((nq, heads, 576), generator=generator).to(torch.bfloat16) * 0.1
        kv = torch.randn((4, 1, 576), generator=generator).to(torch.bfloat16) * 0.1
        indices = (torch.arange(64, dtype=torch.int32) % 4).repeat(nq, 1, 1)
        indices[:, :, ::7] = -1
        return q, kv, indices

    def test_optional_parameters_select_the_correct_specialization(self):
        for use_sink in (False, True):
            for use_lengths in (False, True):
                with self.subTest(sink=use_sink, lengths=use_lengths):
                    q, kv, indices = self.inputs()
                    sink = torch.linspace(-2, 2, 64) if use_sink else None
                    lengths = torch.tensor([64, 0, 3], dtype=torch.int32) if use_lengths else None
                    actual = interface.flash_mla_sparse_fwd(q, kv, indices, 1 / 16, attn_sink=sink, topk_length=lengths)
                    compiled = self.compiles[-1]
                    self.assertEqual(compiled[4:6], (use_sink, use_lengths))
                    args = self.launches[-1]
                    self.assertEqual(args[3].shape, (64,))
                    self.assertEqual(args[4].shape, (3,))
                    self.assertEqual(args[1].data_ptr(), kv.data_ptr())
                    self.assertEqual(args[2].data_ptr(), indices.data_ptr())
                    self.assertIs(actual[0], args[5])
                    if use_lengths:
                        self.assertIs(args[4], lengths)
                        self.assertEqual(torch.count_nonzero(actual[0][1]), 0)
                        self.assertTrue(torch.isneginf(actual[1][1]).all())
                        self.assertTrue(torch.isposinf(actual[2][1]).all())

    def test_h64_out_and_pitched_inputs_are_bound_without_copy(self):
        q = torch.zeros((2, 128, 600), dtype=torch.bfloat16)[:, :64, :576]
        kv = torch.ones((4, 1, 600), dtype=torch.bfloat16)[:, :, :576]
        indices = torch.zeros((2, 1, 80), dtype=torch.int32)[:, :, :64]
        backing = torch.full((2, 128, 520), -9, dtype=torch.bfloat16)
        out = backing[:, :64, :512]
        actual = interface.flash_mla_sparse_fwd(q, kv, indices, 1 / 16, out=out)
        args = self.launches[-1]
        self.assertIs(args[0], q)
        self.assertIs(args[5], out)
        self.assertIs(actual[0], out)
        self.assertEqual(self.compiles[-1][6:10], (tuple(q.stride()), 600, 80, tuple(out.stride())))
        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
        self.assertTrue((backing[:, 64:] == -9).all())
        self.assertTrue((backing[:, :64, 512:] == -9).all())

    def test_small_heads_pad_query_and_sink_then_preserve_out_identity(self):
        for heads in (8, 16, 32):
            for use_sink in (False, True):
                with self.subTest(heads=heads, sink=use_sink):
                    q, kv, indices = self.inputs(heads=heads)
                    sink = torch.zeros(heads) if use_sink else None
                    out = torch.empty((3, heads, 512), dtype=torch.bfloat16)
                    actual = interface.flash_mla_sparse_fwd(q, kv, indices, 1 / 16, attn_sink=sink, out=out)
                    args = self.launches[-1]
                    self.assertEqual(args[0].shape, (3, 64, 576))
                    torch.testing.assert_close(args[0][:, :heads], q)
                    self.assertEqual(torch.count_nonzero(args[0][:, heads:]), 0)
                    if use_sink:
                        self.assertTrue(torch.isneginf(args[3][heads:]).all())
                    self.assertIs(actual[0], out)
                    self.assertIsNot(args[5], out)
                    torch.testing.assert_close(out, args[5][:, :heads])
                    self.assertEqual(actual[1].shape, (3, heads))
                    self.assertEqual(actual[2].shape, (3, heads))

    def test_sink_infinities_keep_statistics_and_have_defined_outputs(self):
        q = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
        kv = torch.ones((1, 1, 576), dtype=torch.bfloat16)
        indices = torch.zeros((1, 1, 64), dtype=torch.int32)
        sink = torch.zeros(64)
        sink[0], sink[1] = -torch.inf, torch.inf
        out, maximum, lse = interface.flash_mla_sparse_fwd(q, kv, indices, 1 / 16, attn_sink=sink)
        torch.testing.assert_close(out[0, 0], torch.ones_like(out[0, 0]))
        self.assertEqual(torch.count_nonzero(out[0, 1]), 0)
        self.assertTrue((maximum == 0).all())
        torch.testing.assert_close(lse, torch.full_like(lse, torch.log(torch.tensor(64.0))))

    def test_empty_queries_do_not_compile_or_launch(self):
        q = torch.empty((0, 8, 576), dtype=torch.bfloat16)
        kv = torch.empty((0, 1, 576), dtype=torch.bfloat16)
        indices = torch.empty((0, 1, 64), dtype=torch.int32)
        out = torch.empty((0, 8, 512), dtype=torch.bfloat16)
        actual = interface.flash_mla_sparse_fwd(q, kv, indices, 1 / 16, out=out)
        self.assertIs(actual[0], out)
        self.assertEqual(actual[1].shape, (0, 8))
        self.compiler.assert_not_called()

    def test_output_is_not_shared_between_calls(self):
        q, kv, indices = self.inputs()
        first = interface.flash_mla_sparse_fwd(q, kv, indices, 1 / 16)[0]
        saved = first.clone()
        second = interface.flash_mla_sparse_fwd(q, kv * 2, indices, 1 / 16)[0]
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, saved)

    def test_invalid_metadata_is_rejected_before_compilation(self):
        q, kv, indices = self.inputs()
        cases = [
            ((q.float(), kv, indices, 1 / 16), {}, "dtype"),
            ((q[:, :0], kv, indices, 1 / 16), {}, "H"),
            ((q, kv.squeeze(1), indices, 1 / 16), {}, "kv"),
            ((q, kv, indices.squeeze(1), 1 / 16), {}, "indices"),
            ((q, kv, indices[:, :, :63], 1 / 16), {}, "topk"),
            ((q, kv, indices, 0), {}, "sm_scale"),
            ((q, kv, indices, 1 / 16), {"d_v": 256}, "d_v"),
            ((q, kv, indices, 1 / 16), {"attn_sink": torch.zeros(64, dtype=torch.bfloat16)}, "dtype"),
            ((q, kv, indices, 1 / 16), {"topk_length": torch.zeros(3)}, "dtype"),
            ((q, kv, indices, 1 / 16), {"out": q[..., :512]}, "overlap"),
            ((q.expand(2, -1, -1) if q.shape[0] == 1 else q[:1].expand(3, -1, -1), kv, indices, 1 / 16), {}, "strides"),
            ((q, torch.empty_like(kv, device="meta"), indices, 1 / 16), {}, "must be on"),
            ((torch.empty((3, 64, 1152), dtype=torch.bfloat16)[..., ::2], kv, indices, 1 / 16), {}, "stride 1"),
        ]
        for args, kwargs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex((ValueError, TypeError), message):
                interface.flash_mla_sparse_fwd(*args, **kwargs)
        self.compiler.assert_not_called()


class CompilationBindingTests(unittest.TestCase):
    def test_explicit_outputs_and_strides_reach_the_compiler(self):
        fake_tilelang = types.ModuleType("tilelang")
        fake_tilelang.compile = Mock(return_value="compiled")
        fake_kernel = types.ModuleType("examples.ascend.hy4_mla.kernel")
        fake_kernel.hy4_sparse_attention = Mock(return_value="program")
        with patch.dict(sys.modules, {"tilelang": fake_tilelang, "examples.ascend.hy4_mla.kernel": fake_kernel}):
            result = compile_bf16(
                3,
                4,
                64,
                scale=1 / 16,
                num_cores=28,
                variable_lengths=True,
                explicit_outputs=True,
                q_strides=(40000, 600, 1),
                kv_stride=600,
                index_stride=80,
                out_strides=(70000, 520, 1),
            )
            self.assertEqual(result, "compiled")
            self.assertEqual(fake_tilelang.compile.call_args.kwargs["out_idx"], [])
            self.assertEqual(fake_kernel.hy4_sparse_attention.call_args.kwargs["kv_stride"], 600)
            self.assertEqual(fake_kernel.hy4_sparse_attention.call_args.kwargs["num_cores"], 28)
            self.assertEqual(fake_kernel.hy4_sparse_attention.call_args.kwargs["out_strides"], (70000, 520, 1))
            compile_bf16(3, 4, 64, scale=1 / 16, num_cores=28)
            self.assertEqual(fake_tilelang.compile.call_args.kwargs["out_idx"], [5, 6, 7])

    def test_strided_automatic_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit_outputs"):
            compile_bf16(3, 4, 64, scale=1 / 16, out_strides=(70000, 520, 1))


if __name__ == "__main__":
    unittest.main()
