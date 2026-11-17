"""Conservative worker-count rewriting for independent grid-stride task loops.

This runs before scheduling, synchronization insertion and constant folding.
It never edits a launch extent without also rebuilding the task enumeration.
"""
from __future__ import annotations

from tvm import arith, tirx
from tvm.ir import structural_equal
from tvm.tirx.stmt_functor import ir_transform, post_order_visit, pre_order_visit, substitute
from tvm.tirx.transform import prim_func_pass

from tilelang.ascend.block_policy import bind_block_policy


def _inline_schedule_aliases(stmt):
    """Expose integer scheduling aliases without inlining VF registers/calls."""
    aliases = {}

    def rewrite(node):
        if isinstance(node, tirx.SBlock) and node.name_hint in ("SIMT_VF", "SIMD_VF"):
            return substitute(node, aliases)
        if isinstance(node, tirx.Bind):
            value = substitute(node.value, aliases)
            dtype = str(node.var.dtype)
            if dtype.startswith(("int", "uint", "bool")) and "x" not in dtype and not _nodes(value, (tirx.Call, tirx.BufferLoad)):
                aliases[node.var] = value
                return tirx.Evaluate(0)
            return tirx.Bind(node.var, value)
        if isinstance(node, tirx.Var):
            return aliases.get(node)
        return None

    return ir_transform(stmt, rewrite, None)


def _nodes(stmt, cls):
    result = []
    post_order_visit(stmt, lambda node: result.append(node) if isinstance(node, cls) else None)
    return result


def _uses(stmt, var):
    return any(node.same_as(var) for node in _nodes(stmt, tirx.Var))


def _integer(expr):
    if isinstance(expr, int):
        return expr
    return int(expr.value) if isinstance(expr, tirx.IntImm) else None


def _replace(stmt, old, new):
    return ir_transform(stmt, lambda node: new if structural_equal(node, old) else None, None)


def _unsafe_calls(stmt):
    for call in _nodes(stmt, tirx.Call):
        name = str(getattr(call.op, "name", ""))
        if not name or any(word in name.lower() for word in (
            "atomic", "barrier", "sync", "set_flag", "wait_flag", "semaphore", "call_extern", "call_packed",
        )):
            return True
    return False


def _resource(stmt):
    # Cube participation determines the budget for an AIC/AIV cooperation unit.
    if any("gemm" in str(getattr(call.op, "name", "")) for call in _nodes(stmt, tirx.Call)):
        return "aic"
    if any(block.name_hint in ("SIMT_VF", "SIMD_VF") for block in _nodes(stmt, tirx.SBlock)):
        return "aiv"
    return None


def _fresh_task_state(stmt):
    """Require scratch reads to be initialized within this task, not a prior task."""
    analyzer = arith.Analyzer()
    initialized = set()
    aliases = {}

    def local(buffer):
        return buffer.scope() not in ("", "global")

    def full(buffer, indices, extents):
        return len(indices) == len(buffer.shape) == len(extents) and all(
            analyzer.can_prove_equal(index, 0) and analyzer.can_prove_equal(extent, size)
            for index, extent, size in zip(indices, extents, buffer.shape))

    def region(arg):
        if isinstance(arg, tirx.Call) and str(getattr(arg.op, "name", "")) == "tl.region":
            load = arg.args[0]
            return load.buffer, list(load.indices), list(arg.args[2:]), _integer(arg.args[1])
        return None

    def expr(node):
        if isinstance(node, tirx.BufferLoad):
            if local(node.buffer) and node.buffer.data not in initialized:
                raise ValueError("scratch read depends on a preceding task")
            for index in node.indices:
                expr(index)
        elif isinstance(node, tirx.Call):
            name = str(getattr(node.op, "name", ""))
            if name == "tl.tileop.gemm":
                for arg in node.args[:2]:
                    expr(arg)
                output = region(node.args[2])
                if output is None:
                    raise ValueError("unknown GEMM output")
                buffer, indices, extents, _ = output
                if not analyzer.can_prove_equal(node.args[9], 1) and buffer.data not in initialized:
                    raise ValueError("GEMM accumulator is not reset for the task")
                if full(buffer, indices, extents):
                    initialized.add(buffer.data)
                return
            access = region(node)
            if name == "tl.access_ptr":
                load = node.args[0]
                if not isinstance(load, tirx.BufferLoad) or len(load.indices) != 1:
                    raise ValueError("unsupported pointer access")
                access = load.buffer, list(load.indices), [node.args[1]], _integer(node.args[2])
            if access:
                buffer, indices, extents, mode = access
                if mode is None:
                    raise ValueError("unknown access mode")
                if mode & 1 and local(buffer) and buffer.data not in initialized:
                    raise ValueError("scratch input is not reset for the task")
                if mode & 2 and full(buffer, indices, extents):
                    initialized.add(buffer.data)
                return
            for arg in node.args:
                expr(arg)
        else:
            # Traverse arithmetic and predicate expressions without interpreting them.
            def read(child):
                if isinstance(child, (tirx.BufferLoad, tirx.Call)):
                    expr(child)
                    return False
                return True
            if isinstance(node, tirx.PrimExpr):
                pre_order_visit(tirx.Evaluate(node), read)

    def walk(node):
        nonlocal initialized
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                walk(child)
        elif isinstance(node, tirx.IfThenElse):
            expr(node.condition)
            before = set(initialized)
            walk(node.then_case)
            then = set(initialized)
            initialized = before
            if node.else_case is not None:
                walk(node.else_case)
            initialized &= then
        elif isinstance(node, tirx.For):
            if _integer(node.min) != 0 or not _integer(node.extent):
                raise ValueError("unknown scratch loop bounds")
            # First iteration must work with fresh state; subsequent iterations
            # may consume state established within this same task (e.g. GEMM K).
            walk(substitute(node.body, {node.loop_var: node.min}))
            walk(node.body)
            for store in _nodes(node.body, tirx.BufferStore):
                if (len(store.indices) == len(store.buffer.shape) == 1
                        and analyzer.can_prove_equal(store.indices[0], node.loop_var)
                        and analyzer.can_prove_equal(node.extent, store.buffer.shape[0])):
                    initialized.add(store.buffer.data)
            # Explicit SIMD contiguous vector stores covering the full buffer.
            for call in _nodes(node.body, tirx.Call):
                if str(getattr(call.op, "name", "")) != "tl.simd.vsts":
                    continue
                ptr, mask = call.args[0], call.args[2]
                if isinstance(mask, tirx.Var):
                    mask = aliases.get(mask, mask)
                if not (isinstance(ptr, tirx.Call) and str(getattr(ptr.op, "name", "")) == "tl.access_ptr"
                        and isinstance(mask, tirx.Call) and str(getattr(mask.op, "name", "")) == "tl.simd.pset"
                        and str(getattr(mask.args[1], "value", "")) == "PAT_ALL"):
                    continue
                load, width = ptr.args[0], ptr.args[1]
                if (isinstance(load, tirx.BufferLoad) and len(load.indices) == len(load.buffer.shape) == 1
                        and analyzer.can_prove_equal(load.indices[0], node.loop_var * width)
                        and analyzer.can_prove_equal(node.extent * width, load.buffer.shape[0])):
                    initialized.add(load.buffer.data)
        elif isinstance(node, tirx.BufferStore):
            expr(node.value)
            for index in node.indices:
                expr(index)
            if full(node.buffer, node.indices, [1] * len(node.indices)):
                initialized.add(node.buffer.data)
        elif isinstance(node, tirx.Evaluate):
            expr(node.value)
        elif isinstance(node, tirx.Bind):
            expr(node.value)
            aliases[node.var] = node.value
        elif isinstance(node, tirx.SBlockRealize):
            walk(node.block)
        elif isinstance(node, tirx.SBlock):
            if node.init is not None:
                walk(node.init)
            walk(node.body)
        elif isinstance(node, tirx.AttrStmt):
            walk(node.body)
        elif not isinstance(node, (tirx.AllocBuffer, tirx.DeclBuffer)):
            raise ValueError("unsupported task state")

    try:
        walk(stmt)
        return True
    except ValueError:
        return False


def _independent_regions(stmt, task, persistent):
    """Reject GM feedback and unchecked stores; check canonical vector footprints.

    Persistent loops already express a task-domain schedule. For unannotated
    loops require a 1-D nonoverlapping output region of exactly one task tile.
    """
    analyzer = arith.Analyzer()
    writes = set()
    dependent = set()
    dependent_vars = set()
    def depends(expr):
        return (_uses(expr, task) or any(var in dependent_vars for var in _nodes(expr, tirx.Var))
                or any(load.buffer.data in dependent for load in _nodes(expr, tirx.BufferLoad)))
    # Scalar scratch often carries swizzled tile coordinates. Track those
    # definitions as well as direct uses of the task index.
    stores = _nodes(stmt, tirx.BufferStore)
    binds = _nodes(stmt, tirx.Bind)
    for _ in range(len(stores) + len(binds) + 1):
        before = len(dependent) + len(dependent_vars)
        for store in stores:
            if depends(store.value):
                dependent.add(store.buffer.data)
        for bind in binds:
            if depends(bind.value):
                dependent_vars.add(bind.var)
        for branch in _nodes(stmt, tirx.IfThenElse):
            if depends(branch.condition):
                dependent.update(store.buffer.data for store in _nodes(branch, tirx.BufferStore))
        if len(dependent) + len(dependent_vars) == before:
            break
    for store in _nodes(stmt, tirx.BufferStore):
        if store.buffer.scope() in ("", "global"):
            return False
    for call in _nodes(stmt, tirx.Call):
        if str(getattr(call.op, "name", "")) != "tl.region":
            continue
        if len(call.args) < 3 or not isinstance(call.args[0], tirx.BufferLoad):
            return False
        load, mode = call.args[0], _integer(call.args[1])
        if load.buffer.scope() not in ("", "global"):
            continue
        # Region BufferLoads denote addresses, not actual reads.
        if mode == 2:
            writes.add(load.buffer.data)
            if persistent and not any(depends(index) for index in load.indices):
                return False
            if not persistent:
                if len(load.indices) != 1 or len(call.args) != 3:
                    return False
                width = _integer(call.args[2])
                if not width or not analyzer.can_prove_equal(load.indices[0], task * width):
                    return False
        elif mode == 1:
            pass
        else:
            return False
    # Recompute reads excluding address-only output region BufferLoads.
    actual_reads = set()
    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == "tl.region":
            if _integer(node.args[1]) == 1:
                load = node.args[0]
                if isinstance(load, tirx.BufferLoad) and load.buffer.scope() in ("", "global"):
                    actual_reads.add(load.buffer.data)
            return False
        if isinstance(node, tirx.BufferLoad) and node.buffer.scope() in ("", "global"):
            actual_reads.add(node.buffer.data)
        return True
    pre_order_visit(stmt, visit)
    return bool(writes) and not (writes & actual_reads)


def _rewrite_grid(grid, budgets):
    if grid.attr_key != "thread_extent" or getattr(grid.node, "thread_tag", "") != "blockIdx.x":
        return grid
    old = _integer(grid.value)
    if not old or _unsafe_calls(grid.body):
        return grid
    # Multi-dimensional or explicitly partitioned mixed launches need their
    # own proof of ownership; leave them untouched.
    if any(attr.attr_key == "thread_extent" and getattr(attr.node, "thread_tag", "") in
           ("blockIdx.y", "blockIdx.z", "cthread") for attr in _nodes(grid.body, tirx.AttrStmt)):
        return grid
    resource = _resource(grid.body)
    cores = budgets.get(resource, 0)
    if not cores:
        return grid
    bx = grid.node.var
    analyzer = arith.Analyzer()
    candidates = []
    for loop in _nodes(grid.body, tirx.For):
        if any(str(key) not in ("num_stages", "tl.persistent_tasks", "tl.persistent_stride")
               for key in loop.annotations):
            continue
        if loop.kind != tirx.ForKind.SERIAL or _integer(loop.min) != 0 or not _integer(loop.extent):
            continue
        if loop.step is not None and _integer(loop.step) != 1:
            continue
        linear = loop.loop_var * old + bx
        # Only exact grid-stride enumeration is accepted, never contiguous
        # per-block chunks such as Attention's bx * tiles_per_block + step.
        matches = [node for node in _nodes(loop.body, tirx.Add)
                   if _uses(node, bx) and _uses(node, loop.loop_var)
                   and analyzer.can_prove_equal(node, linear)]
        if not matches:
            continue
        task = tirx.Var("auto_block_task", loop.loop_var.dtype)
        abstract = loop.body
        for match in matches:
            abstract = _replace(abstract, match, task)
        if _uses(abstract, bx) or _uses(abstract, loop.loop_var):
            continue
        while isinstance(abstract, tirx.IfThenElse) and _integer(abstract.condition) == 1 and abstract.else_case is None:
            abstract = abstract.then_case
        persistent = "tl.persistent_tasks" in loop.annotations
        if persistent:
            tasks = _integer(loop.annotations["tl.persistent_tasks"])
        elif isinstance(abstract, tirx.IfThenElse) and abstract.else_case is None:
            cond = abstract.condition
            tasks = _integer(cond.b) if isinstance(cond, tirx.LT) and analyzer.can_prove_equal(cond.a, task) else None
        else:
            tasks = old * _integer(loop.extent)
        if not tasks or tasks <= 0:
            continue
        # Verify that the old grid enumerated this domain exactly once.
        if persistent:
            # 1-D Persistent may pad its domain for grouping. Recover the old
            # padded bound instead of dropping valid grouped tail tasks.
            padded = [int(cond.b.value) for cond in _nodes(abstract, tirx.LT)
                      if isinstance(cond.b, tirx.IntImm) and analyzer.can_prove_equal(cond.a, task)]
            scheduled = max([tasks, *padded])
        else:
            scheduled = tasks
            if (tasks + old - 1) // old != _integer(loop.extent):
                continue
        if not _independent_regions(abstract, task, persistent):
            continue
        if not _fresh_task_state(abstract):
            continue
        # No block-specific computation or output is permitted outside the
        # task loop. Allocations and kernel-wide mode settings are preserved.
        outside = _replace(grid.body, loop, tirx.Evaluate(0))
        if _uses(outside, bx) or _nodes(outside, tirx.BufferStore):
            continue
        if any("copy" in str(getattr(call.op, "name", "")) or "gemm" in str(getattr(call.op, "name", ""))
               for call in _nodes(outside, tirx.Call)):
            continue
        blocks = min(cores, tasks)
        if blocks == old:
            continue
        body = substitute(abstract, {task: loop.loop_var * blocks + bx})
        # For exact old grids, create the new nondivisible tail guard around
        # the complete task, including its producer/consumer operations.
        if not persistent and not isinstance(abstract, tirx.IfThenElse) and tasks % blocks:
            body = tirx.IfThenElse(loop.loop_var * blocks + bx < tasks, body, None)
        annotations = {str(k): v for k, v in loop.annotations.items()
                       if not str(k).startswith("tl.persistent_")}
        rebuilt = tirx.For(loop.loop_var, 0, (scheduled + blocks - 1) // blocks,
                           loop.kind, body, loop.thread_binding, annotations, step=loop.step)
        candidates.append((loop, rebuilt, blocks))
    if len(candidates) != 1:
        return grid
    old_loop, new_loop, blocks = candidates[0]
    return tirx.AttrStmt(grid.node, grid.attr_key, blocks, _replace(grid.body, old_loop, new_loop))


def AutoBlocks():
    """Adapt eligible task loops using frozen current-device core budgets."""
    @prim_func_pass(opt_level=0, name="tl.ascend.AutoBlocks")
    def transform(func, mod, ctx):
        func = bind_block_policy(func)
        if not int(func.attrs.get("tl.ascend.auto_blocks", 0)):
            return func
        budgets = {unit: int(func.attrs.get(f"tl.ascend.{unit}_cores", 0)) for unit in ("aic", "aiv")}
        if not any(budgets.values()):
            return func
        # VF call bindings have semantic codegen roles; preserve them while
        # exposing only side-effect-free integer scheduling expressions.
        normalized = func.with_body(_inline_schedule_aliases(func.body))
        changed = False
        def rewrite(node):
            nonlocal changed
            if isinstance(node, tirx.AttrStmt):
                updated = _rewrite_grid(node, budgets)
                changed |= not updated.same_as(node)
                return updated
            return None
        body = ir_transform(normalized.body, None, rewrite)
        return normalized.with_body(body) if changed else func
    return transform
