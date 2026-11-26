"""Profiler export parsing, units, target identity and missing-data handling."""
import csv
from pathlib import Path
import struct
import tempfile
import unittest

from profile_metrics import parse


class ProfileMetricsTests(unittest.TestCase):
    def write_csv(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_named_kernel_medians_units_and_no_inferred_time(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            self.write_csv(p / 'op_summary_0.csv', [
                {'Kernel Name': name, 'Task Duration(us)': time,
                 'aiv_vec_ratio': vector, 'MTE2 Ratio (%)': mte,
                 'aiv_mte3_time(us)': 0.25}
                for name, time, vector, mte in [('main_kernel', 4, .5, 20),
                                               ('main_kernel', 6, .7, 40),
                                               ('unrelated', 900, .9, 90)]])
            r = parse(p)
            self.assertEqual(r['status'], 'parsed')
            self.assertEqual(r['samples'], 2)
            self.assertEqual(r['profile_kernel_us'], 5)
            self.assertAlmostEqual(r['vector_ratio'], .6)
            self.assertAlmostEqual(r['mte2_ratio'], .3)
            self.assertEqual(r['mte3_time_us'], .25)
            self.assertEqual(r['vector_time_us'], 'unknown')
            self.assertEqual(r['mte2_time_us'], 'unknown')

    def test_duplicate_exports_are_ambiguous_and_unknown_is_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            row = {'Name': 'main_kernel', 'Duration(us)': 'N/A', 'aiv_vec_ratio': 'nan'}
            self.write_csv(p / 'kernel_details.csv', [row])
            r = parse(p)
            self.assertEqual(r['profile_kernel_us'], 'unknown')
            self.assertEqual(r['vector_ratio'], 'unknown')
            self.write_csv(p / 'kernel_details_2.csv', [row])
            self.assertEqual(parse(p)['status'], 'ambiguous_kernel_tables')

    def test_no_target_does_not_attribute_other_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            self.write_csv(p / 'op_summary.csv', [{'Name': 'other', 'Duration(us)': 123}])
            r = parse(p)
            self.assertEqual(r['status'], 'target_kernel_not_identified')
            self.assertEqual(r['profile_kernel_us'], 'unknown')

    def test_raw_fallback_uses_resolved_name_and_aiv_only(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'hash_dic.slice_0').write_text('123:main_kernel\n')
            track = [0] * 8
            track[3], track[5] = 1 << 32, 123
            (p / 'task_track.slice_0').write_bytes(struct.pack('<8q', *track))
            record = [0] * 16
            record[0], record[1], record[2] = 1 << 32, 100, 1 << 31
            record[4], record[8], record[9] = 50, 20, 10
            record[14], record[15] = 1000, 6000
            (p / 'ffts_profile.slice_0').write_bytes(struct.pack('<16q', *record))
            r = parse(p)
            self.assertEqual(r['status'], 'parsed_raw')
            self.assertEqual(r['profile_kernel_us'], 5)
            self.assertEqual(r['vector_ratio'], .5)
            self.assertEqual(r['mte2_ratio'], .2)
            self.assertEqual(r['mte2_time_us'], 'unknown')
            (p / 'ffts_profile.slice_0').write_bytes(b'bad')
            self.assertEqual(parse(p)['status'], 'unsupported_raw_records')


if __name__ == '__main__':
    unittest.main()
