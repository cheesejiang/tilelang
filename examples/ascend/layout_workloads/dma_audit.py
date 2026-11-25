"""Describe static lowered GM/UB DMA sites, not measured DMA execution cost."""


def collect_dma_descriptors(mod):
    if mod is None:
        return {"status": "missing_lowered_snapshot"}
    from tilelang import tvm
    from tvm import tirx

    analyzer = tvm.arith.Analyzer()
    records = []

    def integer(expr):
        value = analyzer.simplify(expr)
        return int(value) if isinstance(value, tirx.IntImm) else str(value)

    def inspect(node):
        if not isinstance(node, tirx.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        name = node.op.name
        if name not in ("tl.ascend_copy_gm_to_ubuf", "tl.ascend_copy_ubuf_to_gm"):
            return
        args = node.args
        if len(args) not in (8, 11):
            records.append({"op": name, "status": "unknown_signature", "call": str(node)})
            return
        records.append({"op": name, "status": "known", "burst_count": integer(args[3]),
                        "burst_bytes": integer(args[4]), "gm_pitch_bytes": integer(args[-2]),
                        "ub_pitch_bytes": integer(args[-1]), "call": str(node)})

    for function in mod.functions.values():
        if isinstance(function, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(function.body, inspect)
    return {"status": "known" if records else "no_matching_descriptors",
            "scope": "static_descriptor_sites_not_dynamic_counts_or_measured_cost",
            "descriptors": records}
