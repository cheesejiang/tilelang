"""Block policy legality and original-versus-candidate reporting."""
import unittest
import tempfile
from pathlib import Path

from block_config import select_blocks, select_attention_blocks, policies_for_case
from block_tuning import summarize, write_markdown


class BlockConfigTests(unittest.TestCase):
    def test_attention_uses_legal_fixed16_without_changing_tiles(self):
        self.assertEqual(select_blocks("fixed-16", 32, 28, 32), 16)
        self.assertEqual(policies_for_case("mha_bf16", "full"), ("original", "auto", "fixed-16"))
        self.assertEqual(policies_for_case("mha_bf16", "smoke"), ("original", "auto"))
        for tasks, expected in ((4, 4), (32, 16), (2048, 16), (56, 28), (31, 1)):
            self.assertEqual(select_attention_blocks("auto", tasks, 28, tasks), expected)
        self.assertEqual(select_attention_blocks("original", 32, 28, 32), 32)
        with self.assertRaises(ValueError):
            select_attention_blocks("fixed-16", 4, 28, 4)
        for case in ("gqa", "gqa_manual"):
            for profile, tasks in (("smoke", 32), ("full", 2048)):
                self.assertEqual(tasks % 16, 0)
                self.assertEqual(policies_for_case(case, profile), ("original", "auto", "fixed-16"))

    def test_markdown_pairs_profiles_and_preserves_failures(self):
        common = dict(case="vecadd_simt", device_name="9579", status="passed",
                      comparison_status="valid", speedup_vs_original=1.25, speedup=0.75)
        rows = [dict(common, profile="smoke", block_policy="fixed-28", blocks=28, tilelang_latency_us=8),
                dict(common, profile="full", block_policy="fixed-28", blocks=28, tilelang_latency_us=80),
                dict(common, profile="smoke", block_policy="fixed-32", status="failed", blocks=32)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            write_markdown(path, rows)
            report = path.read_text()
        self.assertIn("| fixed-28 | 28 | 8.000 | 1.250× | 0.750× | 80.000 | 1.250× | 0.750× |", report)
        self.assertIn("| fixed-32 | 32 / — | failed | — | — | 未运行 | 未运行 | 未运行 |", report)

    def test_small_workloads_clamp_only_auto_and_preserve_original(self):
        self.assertEqual(select_blocks("original", 32, 28, 16), 32)
        self.assertEqual(select_blocks("auto", 32, 28, 16), 16)
        self.assertEqual(select_blocks("auto", 32, 28, 1024), 28)
        self.assertEqual(select_blocks("c", 32, 28, 16), 28)
        self.assertEqual(select_blocks("c-1", 64, 1, 128), 1)
        self.assertEqual(select_blocks("c+1", 32, 28, 16), 29)
        self.assertEqual(select_blocks("2c", 32, 28, 16), 56)
        with self.assertRaises(ValueError):
            select_blocks("auto", 32, 0, 16)

    def test_self_speedup_is_not_eager_speedup_and_failures_remain(self):
        common = dict(case="gemm_bf16", profile="smoke", status="passed", device_name="9579",
                      shape="[1024,1024,1024]", dtype="bfloat16", timing="profiler")
        rows = [dict(common, block_policy="original", tilelang_latency_us=20, speedup=0.5),
                dict(common, block_policy="auto", tilelang_latency_us=10, speedup=0.8),
                dict(common, block_policy="c", status="failed"),
                dict(common, block_policy="2c", tilelang_latency_us=8, device_name="other")]
        result = summarize(rows)
        self.assertEqual(result[1]["speedup_vs_original"], 2)
        self.assertEqual(result[1]["speedup"], 0.8)
        self.assertNotIn("speedup_vs_original", result[2])
        self.assertNotIn("speedup_vs_original", result[3])
        rows[0]["status"] = "failed"
        self.assertNotIn("speedup_vs_original", summarize(rows)[1])


if __name__ == "__main__":
    unittest.main()
