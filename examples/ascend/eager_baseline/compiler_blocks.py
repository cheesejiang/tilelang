"""Compare the public AutoBlocks Pass off/on for every eager-baseline case."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

from compare import CASES
from block_tuning import write_csv


def summarize(rows):
    lookup = {(row["case"], row["profile"], row["compiler_auto_blocks"]): row for row in rows}
    result = []
    for case, profile in dict.fromkeys((row["case"], row["profile"]) for row in rows):
        before = lookup.get((case, profile, "off"), {})
        after = lookup.get((case, profile, "on"), {})
        item = dict(case=case, profile=profile, device_name=after.get("device_name", before.get("device_name", "unknown")),
                    status="incomplete_or_failed", before_status=before.get("status", "missing"),
                    after_status=after.get("status", "missing"), before_blocks=before.get("blocks", "unknown"),
                    after_blocks=after.get("blocks", "unknown"), compiler_changed=after.get("compiler_changed", "unknown"),
                    before_directory=before.get("result_directory", ""), after_directory=after.get("result_directory", ""))
        if before.get("status") == after.get("status") == "passed":
            try:
                for key in ("device_name", "shape", "dtype", "timing", "arch"):
                    if before.get(key) != after.get(key):
                        raise ValueError(f"Mismatched {key}")
                old, new = float(before["tilelang_latency_us"]), float(after["tilelang_latency_us"])
                old_s, new_s = float(before["speedup"]), float(after["speedup"])
                if not all(math.isfinite(value) and value > 0 for value in (old, new, old_s, new_s)):
                    raise ValueError("Invalid timing")
                item.update(status="passed", before_us=old, after_us=new, gain=old / new,
                            before_s=old_s, after_s=new_s)
            except (KeyError, ValueError, TypeError) as error:
                item["error"] = str(error)
        result.append(item)
    return result


def write_markdown(path, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# 公共 AutoBlocks Pass 性能验证", "",
             "全部用例保持原始 DSL、输入和精度配置，仅切换公共 Pass。耗时单位为 μs。",
             "收益 = 关闭耗时 / 开启耗时；S = eager 耗时 / TileLang 耗时。收益 > 1 表示优化后更快。",
             "实际改写来自编译记录；开启 Pass 不等于发生改写。编译不计入耗时。",
             "失败、缺失或无法比较项不计为通过；此表不自动认定性能达标。", ""]
    for profile in ("smoke", "full"):
        selected = [row for row in rows if row["profile"] == profile]
        if not selected:
            continue
        lines.extend([f"## {profile}", "",
                      "| 算子 | 型号 | 关闭 blocks | 开启 blocks | 实际改写 | 状态 | 关闭耗时 | 开启耗时 | 收益 | 关闭 S | 开启 S |",
                      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"])
        for row in selected:
            values = [row[key] for key in ("case", "device_name", "before_blocks", "after_blocks", "compiler_changed", "status")]
            for key in ("before_us", "after_us", "gain", "before_s", "after_s"):
                values.append(f"{row[key]:.3f}" + ("×" if key in ("gain", "before_s", "after_s") else "") if key in row else "—")
            lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--profiles", nargs="+", choices=["smoke", "full"], default=["smoke", "full"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("compiler-blocks"))
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.samples) <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("rounds, warmup, samples and timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(case, profile, mode) for case in dict.fromkeys(args.cases)
            for profile in dict.fromkeys(args.profiles) for mode in ("off", "on")]
    random.Random(args.seed).shuffle(jobs)
    (args.output / "experiment.json").write_text(json.dumps(dict(
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        job_order=jobs), indent=2))
    rows = []
    for index, (case, profile, mode) in enumerate(jobs, 1):
        directory = args.output / "runs" / f"{case}-{profile}-{mode}-{time.time_ns()}"
        directory.mkdir(parents=True)
        command = [sys.executable, str(Path(__file__).with_name("compare.py")), "--cases", case,
                   "--profile", profile, "--compiler-auto-blocks", mode, "--timing", "profiler",
                   "--rounds", str(args.rounds), "--warmup", str(args.warmup), "--samples", str(args.samples),
                   "--timeout", str(args.timeout), "--device", str(args.device), "--arch", args.arch,
                   "--seed", str(args.seed), "--output", str(directory)]
        print(f"[{index}/{len(jobs)}] {case} {profile} AutoBlocks={mode}", flush=True)
        with (directory / "driver.log").open("w") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = directory / ("summary.csv" if args.rounds > 1 else "results.csv")
        row = {}
        if path.exists():
            with path.open() as handle:
                row = next(csv.DictReader(handle), {})
        if not row or run.returncode:
            row.update(status="failed", driver_exit=run.returncode)
        row.update(case=case, profile=profile, compiler_auto_blocks=mode, result_directory=str(directory))
        rows.append(row)
        write_csv(args.output / "results.csv", rows)
        summary = summarize(rows)
        write_csv(args.output / "summary.csv", summary)
        write_markdown(args.output / "summary.md", summary)
    print(f"Markdown: {args.output / 'summary.md'}", flush=True)
    if any(row["status"] != "passed" for row in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
