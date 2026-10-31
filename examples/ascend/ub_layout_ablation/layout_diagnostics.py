"""Convert pass annotations to portable reports without importing TileLang."""

COST_FIELDS = ("estimated_latency_cycles", "service_cycles", "instructions", "requests", "entries", "partial")
STORAGE_FIELDS = ("ub_native_storage_bytes", "ub_storage_bytes", "ub_extra_storage_bytes",
                  "estimated_ub_native_allocation_bytes", "estimated_ub_allocation_bytes", "estimated_ub_extra_allocation_bytes")
FINAL_IR_FIELDS = ("final_ir_ub_allocation_bytes", "final_ir_ub_extra_vs_baseline_bytes", "final_ir_ub_status")


def collect_final_ub_allocations(mod):
    """Read allocated UB extents after MergeUBAllocations; never round again.

    Report each function separately. Different entry functions are not resident
    together, so their maximum, not their sum, is the module summary.
    """
    from tilelang import tvm
    from tvm import tirx

    analyzer = tvm.arith.Analyzer()
    functions = []
    for global_var, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        allocations = []
        seen = set()

        def inspect(node):
            buffers = ([node.buffer] if isinstance(node, tirx.AllocBuffer) else
                       list(node.alloc_buffers) if isinstance(node, tirx.SBlock) else [])
            for buffer in buffers:
                if buffer.scope() not in ("shared", "shared.dyn") or buffer.data in seen:
                    continue
                seen.add(buffer.data)
                elements = tirx.IntImm("int64", 1)
                for extent in buffer.shape:
                    elements = elements * extent
                dtype = tvm.DataType(str(buffer.dtype))
                expr = analyzer.simplify((elements * dtype.bits * dtype.lanes + 7) // 8)
                size = int(expr) if isinstance(expr, tirx.IntImm) else None
                allocations.append({"buffer": str(buffer.name), "scope": buffer.scope(),
                                    "dtype": str(buffer.dtype), "shape": [str(x) for x in buffer.shape],
                                    "bytes": size, "bytes_expr": str(expr),
                                    "range_relative_to_buffer_base": [0, size] if size is not None else None})

        tirx.stmt_functor.post_order_visit(func.body, inspect)
        if allocations:
            known = all(a["bytes"] is not None for a in allocations)
            functions.append({"function": str(global_var.name_hint), "allocations": allocations,
                              "ub_allocation_bytes": sum(a["bytes"] for a in allocations) if known else None})
    known = all(f["ub_allocation_bytes"] is not None for f in functions)
    return {"source": "tl.MergeUBAllocations", "status": "known" if known else "dynamic_extent",
            "ub_allocation_bytes": max((f["ub_allocation_bytes"] for f in functions), default=0) if known else None,
            "functions": functions}


def final_ir_fields(report, baseline_report=None):
    allocation = report.get("final_ir_ub", {})
    size = allocation.get("ub_allocation_bytes")
    baseline = (baseline_report or {}).get("final_ir_ub", {}).get("ub_allocation_bytes")
    return {"final_ir_ub_allocation_bytes": size if size is not None else "unknown",
            "final_ir_ub_extra_vs_baseline_bytes": size - baseline if size is not None and baseline is not None else "unknown",
            "final_ir_ub_status": allocation.get("status", "missing_snapshot")}


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
    costs = final
    if final.get("cost_known") == 0:
        scoped = next((r for r in reversed(records) if r.get("stage") == "padding_scored_final"), {})
        if scoped.get("cost_known") == 1:
            costs = scoped
            native = next((r for r in reversed(records) if r.get("stage") == "padding_scored_unpadded"), {})
    schemes = []
    for buffer in final.get("buffers", []):
        if "padding_bytes" in buffer:
            schemes.append(f"{buffer['buffer']}: every {buffer['period_rows']} rows +{buffer['padding_bytes']}B")
    result = {key: costs.get(key, "unknown") for key in COST_FIELDS}
    result.update({key: final.get(key, "unknown") for key in STORAGE_FIELDS})
    result["cost_scope"] = costs.get("cost_scope", "unknown")
    result["padding_scheme"] = "; ".join(schemes) or "none_or_unrecognized"
    for key in COST_FIELDS:
        result[f"unpadded_{key}"] = native.get(key, "unknown")
    result["unmodeled_costs"] = "DMA; address computation; synchronization"
    if costs is not final:
        result["unmodeled_costs"] += "; reducer and other excluded operations"
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
