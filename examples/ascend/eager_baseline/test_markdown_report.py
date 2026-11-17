import tempfile
import unittest
from pathlib import Path

from markdown_report import write_comparison_markdown


class MarkdownReportTests(unittest.TestCase):
    def test_single_and_multi_round_results_and_failed_metrics(self):
        rows = [dict(case="gemm_bf16", device_name="9579", profile="full", status="passed",
                     blocks=28, tilelang_latency_us=10, eager_latency_us=20, speedup=2),
                dict(case="vecadd_simt", status="passed", rounds=3, passed_rounds=3,
                     tilelang_latency_us=8, eager_latency_us=8, speedup=1),
                dict(case="failed|case", status="failed", speedup=99, tilelang_latency_us=1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            write_comparison_markdown(path, rows)
            report = path.read_text()
        self.assertIn("10.000 | 20.000 | 2.000×", report)
        self.assertIn("1.000× | 3 / 3", report)
        self.assertIn("failed\\|case", report)
        self.assertNotIn("99.000", report)


if __name__ == "__main__":
    unittest.main()
