"""Markdown reports for direct eager comparison runs (no device imports)."""
import math


def write_comparison_markdown(path, rows):
    def text(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def metric(row, key, suffix=""):
        if row.get("status") != "passed":
            return "—"
        try:
            value = float(row[key])
            if math.isfinite(value) and value > 0:
                return f"{value:.3f}{suffix}"
        except (KeyError, TypeError, ValueError):
            pass
        return "—"

    lines = ["# TileLang / eager 测试结果", "",
             "耗时单位：μs；S = eager 耗时 / TileLang 耗时。S > 1 表示 TileLang 更快。",
             "blocks 为逻辑 block 数；失败、不支持、未完成项不计为通过。多轮耗时采用脚本既定聚合规则。", "",
             "| 算子 | 型号 | 规格 | 策略 | blocks | 状态 | TileLang 耗时 | eager 耗时 | S | 通过轮数 / 总轮数 |",
             "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    for row in rows:
        rounds = f"{row.get('passed_rounds', '—')} / {row['rounds']}" if 'rounds' in row else "—"
        values = [row.get(key, "—") for key in ("case", "device_name", "profile", "block_policy", "blocks", "status")]
        values.extend([metric(row, "tilelang_latency_us"), metric(row, "eager_latency_us"),
                       metric(row, "speedup", "×"), rounds])
        lines.append("| " + " | ".join(text(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
