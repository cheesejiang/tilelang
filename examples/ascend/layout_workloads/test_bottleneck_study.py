"""Host-only controls keep scalar/vector pairs comparable and recover reports."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import run_bottleneck_study as study


class BottleneckTests(unittest.TestCase):
    def test_scalar_pairs_preserve_mode_period_and_launch(self):
        args = SimpleNamespace(output=Path('/tmp/control'), blocks=32, rounds=3,
                               no_profile=False, profile_repeats=10)
        variants = {v[0]: v for v in study.variants()}
        for vector, scalar in [('baseline', 'baseline-scalar'), ('p8', 'p8-scalar')]:
            a, b = [study.command(args, variants[name]) for name in (vector, scalar)]
            for flag in ('--mode', '--scale-padding-period', '--rows', '--columns', '--blocks'):
                self.assertEqual(a[a.index(flag) + 1], b[b.index(flag) + 1])
            self.assertEqual(b[b.index('--vectorize') + 1], 'disable')
            self.assertIn('--profile', a)
        self.assertEqual(len(set(variants)), 10)

    def test_recovery_keeps_missing_metrics_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder))
            path = args.output / 'p8-scalar/gemm_scale/baseline.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'measured', 'correctness_passed': True,
                                        'latency_us': 4.5, 'entries': 2048}))
            study.summarize(args)
            with (args.output / 'summary.csv').open() as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(row['kernel_time_us'], '4.5')
            self.assertEqual(row['model_entries'], '2048')
            self.assertEqual(row['model_instructions'], 'unknown')
            self.assertEqual(row['profile_directory'], 'not_collected')
            self.assertEqual(row['vector_ratio'], 'unknown')
            profile = args.output / 'saved-profile'
            profile.mkdir()
            (profile / 'op_summary.csv').write_text(
                'Kernel Name,Task Duration(us),aiv_vec_ratio\nmain_kernel,4.8,0.6\n')
            report = json.loads(path.read_text())
            report['profile'] = {'directory': str(profile)}
            path.write_text(json.dumps(report))
            study.summarize(args)
            with (args.output / 'profile_summary.csv').open() as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(row['profile_status'], 'parsed')
            self.assertEqual(row['profile_kernel_us'], '4.8')
            self.assertEqual(row['vector_ratio'], '0.6')
            self.assertEqual(row['vector_time_us'], 'unknown')
            self.assertTrue((profile / 'metrics_summary.json').exists())


if __name__ == '__main__':
    unittest.main()
