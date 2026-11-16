"""Host-side checks of raw interval aggregation and profiler window boundaries."""
import contextlib
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_parser
import profiler_timer


class ProfilerTests(unittest.TestCase):
    def test_overlapping_kernels_use_union_not_sum_or_span(self):
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for start, end in [(1000, 5000), (3000, 7000), (10000, 12000)]:
                values = [0] * 16
                values[1] = 1
                values[14:16] = [start, end]
                records.append(struct.pack('<16q', *values))
            path = Path(folder) / 'ffts_profile.slice_0'
            path.write_bytes(b''.join(records))
            result = profile_parser._parse_ffts_profile(Path(folder), 1)
            self.assertEqual(result['latency_ms'], 0.008)
            self.assertEqual(result['profile_kernel_us'], 10)
            self.assertEqual(result['kernel_span_ms'], 0.011)
            path.write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError, 'Truncated'):
                profile_parser._parse_ffts_profile(Path(folder), 1)

    def test_empty_profile_fails_instead_of_event_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'no ffts_profile'):
                profile_parser._parse_ffts_profile(Path(folder), 1)

    def test_flush_outside_window_and_all_operator_work_inside(self):
        events = []
        timer = object.__new__(profiler_timer.ProfilerTimer)
        timer.directory = Path('/not-written')
        timer.buffer = object()
        timer.flush = lambda buffer: events.append('flush')
        timer.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append('sync')))
        @contextlib.contextmanager
        def profile(directory):
            events.append('enter')
            yield SimpleNamespace(step=lambda: events.append('step'), prof_if=SimpleNamespace(prof_path='trace'))
            events.append('exit')
        good = dict(latency_ms=1, profile_kernel_count=2, profile_excluded_kernel_count=0)
        with patch.object(profiler_timer, 'profile', profile), patch.object(profiler_timer, '_parse_ffts_profile', return_value=good):
            timer.measure(lambda: events.append('operator'), 'tilelang', 0)
        self.assertEqual(events, ['flush', 'sync', 'enter', 'operator', 'sync', 'step', 'exit'])
        bad = dict(good, profile_excluded_kernel_count=1)
        with patch.object(profiler_timer, 'profile', profile), patch.object(profiler_timer, '_parse_ffts_profile', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'cache flush'):
                timer.measure(lambda: None, 'eager', 0)


if __name__ == '__main__':
    unittest.main()
