"""Host-only reader for exported CANN kernel CSVs; never infer phase times."""
import csv
import json
import math
from pathlib import Path
import re
import statistics
import struct
import importlib.util


FIELDS = ("profile_kernel_us", "vector_ratio", "scalar_ratio", "mte1_ratio",
          "mte2_ratio", "mte3_ratio", "vector_time_us", "mte2_time_us", "mte3_time_us")


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def number(value):
    try:
        result = float(str(value).strip().rstrip('%'))
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def field(header):
    name = normalize(header)
    if name in ("taskdurationus", "durationus", "aicoretimeus"):
        return "profile_kernel_us"
    for stem, target in (("vec", "vector"), ("vector", "vector"), ("scalar", "scalar"),
                         ("mte1", "mte1"), ("mte2", "mte2"), ("mte3", "mte3")):
        if name in (f"aiv{stem}ratio", f"{stem}ratio", f"aiv{stem}utilization", f"{stem}utilization"):
            return target + "_ratio"
        if target in ("vector", "mte2", "mte3") and name in (
                f"aiv{stem}timeus", f"{stem}timeus"):
            return target + "_time_us"
    return None


def parse(directory):
    directory = Path(directory)
    result = {"status": "no_exported_kernel_csv", **dict.fromkeys(FIELDS, "unknown"),
              "kernels": [], "warnings": [], "sources": []}
    if not directory.is_dir():
        result["status"] = "missing_directory"
        return result
    # Prefer the CANN op_summary export. Do not sum duplicated exports or
    # combine similarly named columns from different profiler tables.
    paths = sorted(directory.rglob("op_summary*.csv"))
    if not paths:
        paths = sorted(directory.rglob("kernel_details*.csv"))
    tables = []
    for path in paths:
        grouped = {}
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = reader.fieldnames or []
                name_key = next((h for wanted in ("kernelname", "name", "opname")
                                 for h in headers if normalize(h) == wanted), None)
                if name_key is None:
                    result["warnings"].append(f"No kernel name column: {path}")
                    continue
                mapped = [(h, field(h)) for h in headers if field(h)]
                for row in reader:
                    name = row.get(name_key, "").strip()
                    if not name:
                        continue
                    item = grouped.setdefault(name, {"name": name, "rows": 0, "values": {},
                                                     "raw_columns": {}})
                    item["rows"] += 1
                    for h, key in mapped:
                        value = number(row.get(h))
                        if value is None:
                            continue
                        # Bare ratios in CANN exports are fractions. Explicit
                        # '%' headers/values are percentages. Do not guess
                        # scale from magnitude or convert cycles into time.
                        if key.endswith("_ratio") and ('%' in h or '%' in str(row.get(h))):
                            value /= 100
                        item["values"].setdefault(key, []).append(value)
                        item["raw_columns"][key] = h
            for item in grouped.values():
                item["metrics"] = {k: statistics.median(v) for k, v in item.pop("values").items()}
                item["source"] = str(path)
                result["kernels"].append(item)
                if "main_kernel" in item["name"] or "gemm_scale" in item["name"]:
                    tables.append(item)
            result["sources"].append(str(path))
        except (OSError, UnicodeError, csv.Error) as error:
            result["warnings"].append(f"{path}: {error}")
    # Multiple kernel identities/devices/duplicate exports require inspection;
    # never silently attribute unrelated activity to the tested kernel.
    if len(tables) == 1:
        result.update(tables[0]["metrics"])
        result["kernel_name"] = tables[0]["name"]
        result["samples"] = tables[0]["rows"]
        result["status"] = "parsed"
    elif tables:
        result["status"] = "ambiguous_kernel_tables"
    elif result["sources"]:
        result["status"] = "target_kernel_not_identified"
    if not result["sources"]:
        return parse_raw(directory, result)
    return result


def parse_raw(directory, result):
    """Fallback using the CANN 9.2 record layout already used in this repo."""
    paths = sorted(directory.rglob("ffts_profile*"))
    if not paths:
        return result
    helper = Path(__file__).resolve().parents[1] / "eager_baseline/profile_parser.py"
    spec = importlib.util.spec_from_file_location("layout_raw_profile_names", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = module._resolve_kernel_names(directory)
    grouped = {}
    for path in paths:
        if path.name.endswith(".done") or not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) % 128:
            result["warnings"].append(f"Unsupported/truncated raw record: {path}")
            result["status"] = "unsupported_raw_records"
            return result
        for offset in range(0, len(data), 128):
            v = struct.unpack_from("<16q", data, offset)
            if v[1] <= 0:
                continue
            name = names.get((v[0] >> 32) & 65535, "unknown")
            # These workloads are pure AIV kernels; do not mix AIC records.
            if v[2] < 1 << 31 or v[15] < v[14]:
                continue
            metrics = {"profile_kernel_us": (v[15] - v[14]) / 1000,
                       **{key: v[index] / v[1] for key, index in
                          (("vector_ratio", 4), ("scalar_ratio", 6),
                           ("mte1_ratio", 7), ("mte2_ratio", 8), ("mte3_ratio", 9))}}
            grouped.setdefault(name, []).append(metrics)
        result["sources"].append(str(path))
    for name, samples in grouped.items():
        metrics = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        result["kernels"].append({"name": name, "rows": len(samples), "metrics": metrics})
    target = [k for k in result["kernels"] if "main_kernel" in k["name"] or "gemm_scale" in k["name"]]
    result["raw_format"] = "CANN 9.2 ffts_profile 128-byte layout (repository parser)"
    if len(target) == 1:
        result.update(target[0]["metrics"])
        result.update(status="parsed_raw", kernel_name=target[0]["name"], samples=target[0]["rows"])
    else:
        result["status"] = "ambiguous_kernel_tables" if target else "target_kernel_not_identified"
    return result


def save(directory):
    result = parse(directory)
    if Path(directory).is_dir():
        (Path(directory) / "metrics_summary.json").write_text(json.dumps(result, indent=2))
    return result
