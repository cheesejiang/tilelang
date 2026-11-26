"""Disassembly routing and rejection of unavailable or host-only output."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import disassemble_workload as dis


class DisassemblyTests(unittest.TestCase):
    def test_variant_routing_and_unsupported_output(self):
        paths = dict(dis.sources(Path('/results'), ['p8', 'p8-scalar', 'ownership', 'padding-search']))
        self.assertEqual(paths['p8'], Path('/results/p8/gemm_scale/baseline.cpp'))
        self.assertEqual(paths['ownership'], Path('/results/ownership/gemm_scale/ownership.cpp'))
        self.assertEqual(paths['padding-search'], Path('/results/padding-search/gemm_scale/padding.cpp'))
        self.assertFalse(dis.decoded('  0: 0000 <not available>'))
        self.assertFalse(dis.decoded('  0: 0000 <unknown>'))
        self.assertTrue(dis.decoded('  0: 0000 LDS.b64 R0,[R2]'))
        with self.assertRaises(ValueError):
            dis.sources(Path('/results'), ['typo'])

    def test_compiles_exact_source_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            source = p / 'baseline.cpp'
            source.write_text('__ubuf__ float* p;\n')
            args = SimpleNamespace(objdump='objdump', compiler='bisheng', arch='dav-3510',
                                   include=[], compile_arg=[])
            calls = []
            def fake_run(cmd, log):
                calls.append(cmd)
                if '--cce-aicore-only' in cmd:
                    header = bytearray(20)
                    header[:4], header[5] = b'\x7fELF', 1
                    header[18:20] = (999).to_bytes(2, 'little')
                    Path(cmd[cmd.index('-o') + 1]).write_bytes(header)
                text = '  0: 0000 LDS.b64 R0,[R2]\n' if '-d' in cmd else ''
                log.write_text(text)
                return text
            with patch.object(dis.shutil, 'which', side_effect=lambda x: '/tools/' + x), \
                    patch.object(dis, 'run', side_effect=fake_run), redirect_stdout(io.StringIO()):
                self.assertTrue(dis.process(source, p / 'isa', args))
            self.assertEqual((p / 'isa/kernel.asc').read_bytes(), source.read_bytes())
            record = json.loads((p / 'isa/manifest.json').read_text())
            self.assertEqual(record['status'], 'generated_unreviewed')
            self.assertEqual(record['isa_review'], 'pending')
            self.assertIn('not a hash-verified measured binary', record['provenance'])
            self.assertEqual(len(calls), 3)

    def test_host_object_is_not_disassembled_as_npu(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            header = bytearray(20)
            header[:4], header[5] = b'\x7fELF', 1
            header[18:20] = (62).to_bytes(2, 'little')
            obj = p / 'host.o'
            obj.write_bytes(header)
            args = SimpleNamespace(objdump='objdump', arch='dav-3510')
            with patch.object(dis.shutil, 'which', return_value='/tools/objdump'), \
                    redirect_stdout(io.StringIO()):
                self.assertFalse(dis.process(obj, p / 'out', args))
            self.assertIn('Host ELF', json.loads((p / 'out/manifest.json').read_text())['error'])


if __name__ == '__main__':
    unittest.main()
