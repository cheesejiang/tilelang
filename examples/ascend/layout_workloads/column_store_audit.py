"""Inspect actual lowered column-store addresses, without evaluating values."""


def collect_column_store_audit(mod, buffer_name, max_steps=65536):
    """Enumerate static TIR coordinates and masks after AscendLowerTileOp.

    Groups describe scalar TIR store sites per serial coordinate and warp,
    not guaranteed backend machine instructions. Addresses are relative to an
    aligned UB base. Unknown predicates/bindings fail closed; never infer lane
    participation from matrix shape or from the source Parallel loop alone.
    """
    if mod is None:
        return {"status": "missing_lowered_snapshot"}
    from tilelang import tvm
    from tvm import tirx

    analyzer = tvm.arith.Analyzer()
    batches = {}
    steps = 0

    def value(expr, bindings):
        result = analyzer.simplify(tirx.stmt_functor.substitute(expr, bindings))
        if not isinstance(result, tirx.IntImm):
            raise ValueError(f"unresolved coordinate or mask: {result}")
        return int(result)

    def relevant(stmt):
        found = False

        def inspect(node):
            nonlocal found
            if isinstance(node, tirx.BufferStore) and str(node.buffer.name) == buffer_name:
                found = True

        tirx.stmt_functor.post_order_visit(stmt, inspect)
        return found

    def walk(stmt, bindings, serial=(), thread=None):
        nonlocal steps
        steps += 1
        if steps > max_steps:
            raise ValueError("static enumeration budget exceeded")
        if not relevant(stmt):
            return
        if isinstance(stmt, tirx.SBlockRealize):
            if value(stmt.predicate, bindings):
                walk(stmt.block, bindings, serial, thread)
        elif isinstance(stmt, tirx.SBlock):
            walk(stmt.body, bindings, serial, thread)
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                walk(child, bindings, serial, thread)
        elif isinstance(stmt, tirx.AttrStmt):
            if stmt.attr_key == "thread_extent":
                tag = str(stmt.node.thread_tag)
                if tag.startswith("blockIdx"):
                    walk(stmt.body, {**bindings, stmt.node.var: 0}, serial, thread)
                elif tag == "threadIdx.x":
                    for t in range(value(stmt.value, bindings)):
                        walk(stmt.body, {**bindings, stmt.node.var: t}, serial, t)
                else:
                    raise ValueError(f"unsupported thread axis: {tag}")
            else:
                walk(stmt.body, bindings, serial, thread)
        elif isinstance(stmt, tirx.For):
            if stmt.kind == tirx.ForKind.VECTORIZED:
                raise ValueError("vectorized store requires machine-width analysis")
            start, count = value(stmt.min, bindings), value(stmt.extent, bindings)
            if stmt.thread_binding is not None:
                tag = str(stmt.thread_binding.thread_tag)
                if tag.startswith("blockIdx"):
                    walk(stmt.body, {**bindings, stmt.loop_var: start}, serial, thread)
                    return
                if tag == "threadIdx.x":
                    for i in range(start, start + count):
                        walk(stmt.body, {**bindings, stmt.loop_var: i}, serial, i - start)
                    return
                raise ValueError(f"unsupported thread axis: {tag}")
            if count < 0 or count > max_steps:
                raise ValueError("loop domain exceeds audit budget")
            for i in range(start, start + count):
                walk(stmt.body, {**bindings, stmt.loop_var: i}, (*serial, i), thread)
        elif isinstance(stmt, tirx.IfThenElse):
            branch = stmt.then_case if value(stmt.condition, bindings) else stmt.else_case
            if branch is not None:
                walk(branch, bindings, serial, thread)
        elif isinstance(stmt, tirx.BufferStore):
            if thread is None:
                raise ValueError("store has no resolved SIMT thread coordinate")
            buffer = stmt.buffer
            if str(buffer.dtype) != "float32" or buffer.data_alignment <= 0 or buffer.data_alignment % 8:
                raise ValueError("audit requires aligned scalar f32 stores")
            indices = [value(index, bindings) for index in stmt.indices]
            offset = 0
            if len(buffer.strides):
                offset += sum(i * value(s, bindings) for i, s in zip(indices, buffer.strides))
            else:
                for i, extent in zip(indices, buffer.shape):
                    offset = offset * value(extent, bindings) + i
            offset += value(buffer.elem_offset, bindings)
            # Preserve a store site's identity, even for structurally equal
            # expressions. A duplicate lane in a site cannot be silently merged.
            key = (stmt, serial, thread // 32)
            lane = thread % 32
            lanes = batches.setdefault(key, {})
            if lane in lanes:
                raise ValueError("duplicate lane in a scalar store batch")
            lanes[lane] = offset * 4
        else:
            raise ValueError(f"unsupported enclosing statement: {type(stmt).__name__}")

    try:
        for func in mod.functions.values():
            if isinstance(func, tirx.PrimFunc):
                walk(func.body, {})
    except Exception as error:
        # Diagnostic extraction must not prevent the real kernel correctness
        # check. TVM simplification errors are unknown evidence as well.
        return {"status": "unknown", "reason": str(error), "source": "tl.AscendLowerTileOp"}
    if not batches:
        return {"status": "no_matching_stores", "buffer": buffer_name}
    samples = []
    for (_, serial, warp), lanes in batches.items():
        groups = {}
        for address in lanes.values():
            groups.setdefault((address // 8) % 16, set()).add(address // 8)
        samples.append({"serial_coordinates": serial, "warp": warp, "active_lanes": len(lanes),
                        "groups": len(groups), "max_distinct_lines_per_group": max(map(len, groups.values())),
                        "lane_byte_addresses": {str(k): v for k, v in sorted(lanes.items())}})
    return {"status": "known", "source": "tl.AscendLowerTileOp", "buffer": buffer_name,
            "scope": "scalar_TIR_store_batches_not_backend_ISA", "batches": len(samples),
            "min_active_lanes": min(s["active_lanes"] for s in samples),
            "max_active_lanes": max(s["active_lanes"] for s in samples),
            "max_distinct_lines_per_group": max(s["max_distinct_lines_per_group"] for s in samples),
            "conflicting_batches": sum(s["max_distinct_lines_per_group"] > 1 for s in samples),
            "samples": samples[:8]}
