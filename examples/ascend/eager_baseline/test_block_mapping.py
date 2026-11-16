"""Host-only checks for diagnostic insertion and durable record validation."""
import unittest
from diagnose_block_mapping import MAGIC, RECORD_WORDS, decode_records, instrument_source

SOURCE = '''#include <c_api/asc_simd.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* C) {
  asc_init();
  if (block_idx < 2) {
    asc_copy_l0c2gm(C);
  }
}
'''


class MappingTests(unittest.TestCase):
    def test_records_are_added_outside_work_guard_and_count_is_inside(self):
        code = instrument_source(SOURCE, 1024 * 1024)
        self.assertEqual(code.count('++__tl_diag_tiles'), 1)
        self.assertIn('if (block_idx < 2) {\n    ++__tl_diag_tiles;\n    asc_copy_l0c2gm(C);', code)
        self.assertLess(code.index('asc_get_system_cycle'), code.index('if (block_idx < 2)'))
        self.assertGreater(code.index('const int64_t __tl_diag_end'), code.index('asc_copy_l0c2gm(C)'))
        self.assertIn('C + 1048576', code)
        self.assertIn('asc_sync();\n  const int64_t __tl_diag_end', code)
        self.assertEqual(instrument_source(code, 1048576), code)
        with self.assertRaises(ValueError):
            instrument_source(SOURCE.replace('__cube__', '__vector__'), 1048576)

    def test_raw_core_id_is_not_replaced_by_logical_id(self):
        records = []
        for block, core, start, end, tasks in [(0, 42, 1000, 11000, 1),
                                               (1, 42, 12000, 22000, 1),
                                               (2, 73, 1500, 2000, 0)]:
            records.append([MAGIC, block, core, start, end, tasks, core] + [-1] * (RECORD_WORDS - 7))
        rows = decode_records(records, 3, 2, 1000)
        self.assertEqual(rows[0]['physical_core_raw'], rows[1]['physical_core_raw'])
        self.assertNotEqual(rows[0]['physical_core_raw'], rows[0]['logical_block'])
        self.assertEqual(rows[1]['start_us'], 11)
        self.assertEqual(rows[1]['duration_us'], 10)
        self.assertEqual(rows[2]['issued_output_tiles'], 0)

    def test_missing_or_wrong_task_records_fail(self):
        row = [MAGIC, 0, 42, 1000, 2000, 16, 42] + [-1] * (RECORD_WORDS - 7)
        self.assertEqual(decode_records([row], 1, 16, 1000)[0]['duration_us'], 1)
        for index, value in [(0, -1), (1, 99), (4, 999), (5, 15)]:
            broken = row.copy()
            broken[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                decode_records([broken], 1, 16, 1000)
        with self.assertRaises(ValueError):
            decode_records([], 1, 16, 1000)


if __name__ == '__main__':
    unittest.main()
