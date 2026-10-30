"""Convert pass annotations to portable reports without importing TileLang."""

COST_FIELDS = ("estimated_latency_cycles", "service_cycles", "instructions", "requests", "entries", "partial")
STORAGE_FIELDS = ("ub_native_storage_bytes", "ub_storage_bytes", "ub_extra_storage_bytes",
                  "ub_native_allocation_bytes", "ub_allocation_bytes", "ub_extra_allocation_bytes")


def plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if hasattr(value, "value"):
        return plain(value.value)
    try:
        return [plain(v) for v in value]
    except TypeError:
        return str(value)


def final_report(records):
    return next((r for r in reversed(records) if r.get("stage") == "final"), {})


def summary_fields(records):
    final = final_report(records)
    native = next((r for r in reversed(records) if r.get("stage") == "final_ownership_unpadded"), {})
    schemes = []
    for buffer in final.get("buffers", []):
        if "padding_bytes" in buffer:
            schemes.append(f"{buffer['buffer']}: every {buffer['period_rows']} rows +{buffer['padding_bytes']}B")
    result = {key: final.get(key, "unknown") for key in (*COST_FIELDS, *STORAGE_FIELDS)}
    result["padding_scheme"] = "; ".join(schemes) or "none_or_unrecognized"
    for key in COST_FIELDS:
        result[f"unpadded_{key}"] = native.get(key, "unknown")
    result["unmodeled_costs"] = "DMA; address computation; synchronization"
    return result


def candidate_rows(records):
    final = final_report(records)
    chosen = {(b["buffer"], b["period_elements"], b["padding_bytes"]) for b in final.get("buffers", []) if "padding_bytes" in b}
    for record in records:
        if record.get("stage") != "padding_candidate":
            continue
        row = {k: v for k, v in record.items() if k != "buffers"}
        row["selected_final"] = (record.get("buffer"), record.get("period_elements"), record.get("padding_bytes")) in chosen
        yield row
