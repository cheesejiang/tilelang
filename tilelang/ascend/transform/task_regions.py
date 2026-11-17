"""Bounded proof of disjoint static output boxes for grid-stride tasks."""
from tvm import DataType, arith, tirx
from tvm.ir import structural_equal
from tvm.tirx.stmt_functor import post_order_visit


def _nodes(stmt, cls):
    result = []
    post_order_visit(stmt, lambda node: result.append(node) if isinstance(node, cls) else None)
    return result


def _region(call):
    if str(getattr(call.op, "name", "")) == "tl.region" and len(call.args) >= 3:
        load = call.args[0]
        if isinstance(load, tirx.BufferLoad) and load.buffer.scope() in ("", "global") and int(call.args[1]) == 2:
            return load.buffer, list(load.indices), list(call.args[2:])
    return None


def _constant(expr):
    if isinstance(expr, (int, tirx.IntImm)):
        return int(expr)
    raise ValueError("not a static integer")


def prove_disjoint_outputs(stmt, task, tasks):
    """Prove disjoint writes, never infer injectivity from task dependence alone.

    Canonical 1-D outputs have a symbolic fast path. Other static mappings,
    including scalar-buffer swizzles, are enumerated under strict proof budgets.
    Unknown/data-dependent coordinates and exhausted budgets mean 'not proven'.
    """
    analyzer = arith.Analyzer()
    regions = [region for call in _nodes(stmt, tirx.Call) if (region := _region(call)) is not None]
    if not regions:
        return False
    try:
        # The coordinate interpreter models direct scalar stores, not pointer
        # escapes or DMA writes into coordinate buffers. Reject those rather
        # than accidentally reuse a scalar value from before an opaque write.
        scalar_data = {load.buffer.data for load in _nodes(stmt, tirx.BufferLoad) if load.buffer.scope() == "local.var"}
        for call in _nodes(stmt, tirx.Call):
            name = str(getattr(call.op, "name", ""))
            if name in ("tl.region", "tl.access_ptr"):
                load = call.args[0]
                mode = _constant(call.args[1] if name == "tl.region" else call.args[2])
                if isinstance(load, tirx.BufferLoad) and load.buffer.scope() == "local.var" and mode & 2:
                    return False
                continue
            if any(isinstance(arg, tirx.Var) and arg in scalar_data
                   or isinstance(arg, tirx.BufferLoad) and arg.buffer.scope() == "local.var" for arg in call.args):
                return False
        buffers = {}
        for buffer, indices, extents in regions:
            shape = [_constant(size) for size in buffer.shape]
            if any(size <= 0 for size in shape) or len(indices) != len(shape) or len(extents) != len(shape):
                return False
            # Logical boxes imply physical separation only for contiguous
            # buffers. Different views of one data pointer need an alias proof.
            stride = 1
            for axis in reversed(range(len(shape))):
                if buffer.strides and _constant(buffer.strides[axis]) != stride:
                    return False
                stride *= shape[axis]
            if buffer.data in buffers and not structural_equal(buffers[buffer.data], buffer):
                return False
            buffers[buffer.data] = buffer

        # Every write to a given output must use the same task-sized footprint;
        # otherwise two individually injective mappings could overlap each other.
        widths = {}
        fast = True
        for buffer, indices, extents in regions:
            width = _constant(extents[0]) if len(indices) == 1 else 0
            if (not width or width <= 0 or not analyzer.can_prove_equal(indices[0], task * width)
                    or _constant(buffer.shape[0]) < tasks * width
                    or buffer.data in widths and widths[buffer.data] != width):
                fast = False
                break
            dtype = DataType(indices[0].dtype)
            if dtype.type_code not in (0, 1) or tasks * width > (1 << (dtype.bits - (dtype.type_code == 0))):
                fast = False
                break
            widths[buffer.data] = width
        if fast:
            return True
        if tasks > 4096:
            return False

        boxes = {key: [] for key in buffers}
        relevant = {}
        steps = 0
        emitted = 0

        def needed(node):
            if node not in relevant:
                relevant[node] = (any(store.buffer.scope() == "local.var" for store in _nodes(node, tirx.BufferStore))
                                  or any(_region(call) is not None for call in _nodes(node, tirx.Call)))
            return relevant[node]

        def checked(node, value):
            dtype = DataType(node.dtype)
            if str(dtype) == "bool" and value in (0, 1):
                return value
            if dtype.type_code not in (0, 1):
                raise ValueError("noninteger coordinate")
            low = -(1 << (dtype.bits - 1)) if dtype.type_code == 0 else 0
            high = (1 << (dtype.bits - (dtype.type_code == 0))) - 1
            if not low <= value <= high:
                raise ValueError("coordinate arithmetic could overflow")
            return value

        def evaluate(node, values, scalars):
            if isinstance(node, tirx.IntImm):
                return int(node)
            if isinstance(node, tirx.Var):
                value = values.get(node)
                if value is None:
                    raise ValueError("unknown coordinate variable")
                return value
            if isinstance(node, tirx.BufferLoad):
                if node.buffer.scope() != "local.var" or any(evaluate(index, values, scalars) != 0 for index in node.indices):
                    raise ValueError("data-dependent coordinate")
                value = scalars.get(node.buffer.data)
                if value is None:
                    raise ValueError("undefined scalar coordinate")
                return value
            if isinstance(node, tirx.Cast):
                return checked(node, evaluate(node.value, values, scalars))
            if isinstance(node, tirx.Select):
                return evaluate(node.true_value if evaluate(node.condition, values, scalars) else node.false_value, values, scalars)
            if isinstance(node, tirx.Not):
                return int(not evaluate(node.a, values, scalars))
            if not hasattr(node, "a") or not hasattr(node, "b"):
                raise ValueError("unsupported coordinate expression")
            a, b = evaluate(node.a, values, scalars), evaluate(node.b, values, scalars)
            kind = type(node).__name__
            if kind in ("Div", "Mod"):
                quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                value = quotient if kind == "Div" else a - quotient * b
            else:
                operations = {
                    "Add": lambda: a + b, "Sub": lambda: a - b, "Mul": lambda: a * b,
                    "FloorDiv": lambda: a // b, "FloorMod": lambda: a % b,
                    "Min": lambda: min(a, b), "Max": lambda: max(a, b),
                    "LT": lambda: int(a < b), "LE": lambda: int(a <= b),
                    "GT": lambda: int(a > b), "GE": lambda: int(a >= b),
                    "EQ": lambda: int(a == b), "NE": lambda: int(a != b),
                    "And": lambda: int(bool(a) and bool(b)), "Or": lambda: int(bool(a) or bool(b)),
                }
                if kind not in operations:
                    raise ValueError("unsupported coordinate arithmetic")
                value = operations[kind]()
            return checked(node, value)

        def walk(node, task_id, values, scalars):
            nonlocal steps, emitted
            steps += 1
            if steps > 1000000:
                raise ValueError("coordinate proof budget exhausted")
            if isinstance(node, tirx.SeqStmt):
                for child in node.seq:
                    walk(child, task_id, values, scalars)
            elif isinstance(node, tirx.Bind):
                try:
                    values[node.var] = evaluate(node.value, values, scalars)
                except ValueError:
                    values[node.var] = None
            elif isinstance(node, tirx.BufferStore) and node.buffer.scope() == "local.var":
                if any(evaluate(index, values, scalars) != 0 for index in node.indices):
                    raise ValueError("unsupported scalar array")
                scalars[node.buffer.data] = evaluate(node.value, values, scalars)
            elif isinstance(node, tirx.IfThenElse):
                if needed(node):
                    branch = node.then_case if evaluate(node.condition, values, scalars) else node.else_case
                    if branch is not None:
                        walk(branch, task_id, values, scalars)
            elif isinstance(node, tirx.For):
                if needed(node.body):
                    start, extent = _constant(node.min), _constant(node.extent)
                    step = 1 if node.step is None else _constant(node.step)
                    if extent < 0 or extent > 4096 or step != 1:
                        raise ValueError("unsupported coordinate loop")
                    for index in range(start, start + extent):
                        values[node.loop_var] = index
                        walk(node.body, task_id, values, scalars)
            elif isinstance(node, tirx.SBlockRealize):
                if needed(node.block) and evaluate(node.predicate, values, scalars):
                    walk(node.block, task_id, values, scalars)
            elif isinstance(node, (tirx.SBlock, tirx.AttrStmt)):
                if isinstance(node, tirx.SBlock) and node.init is not None:
                    walk(node.init, task_id, values, scalars)
                walk(node.body, task_id, values, scalars)
            elif isinstance(node, tirx.Evaluate):
                for call in _nodes(node.value, tirx.Call):
                    region = _region(call)
                    if region is None:
                        continue
                    buffer, indices, extents = region
                    start = tuple(evaluate(index, values, scalars) for index in indices)
                    end = tuple(lo + evaluate(extent, values, scalars) for lo, extent in zip(start, extents))
                    if any(lo < 0 or hi <= lo or hi > _constant(size) for lo, hi, size in zip(start, end, buffer.shape)):
                        raise ValueError("output region outside its buffer")
                    boxes[buffer.data].append((start, end, task_id))
                    emitted += 1
                    if emitted > 16384:
                        raise ValueError("output proof budget exhausted")

        for task_id in range(tasks):
            walk(stmt, task_id, {task: task_id}, {})
        comparisons = 0
        for regions in boxes.values():
            active = []
            for start, end, task_id in sorted(regions):
                active = [box for box in active if box[1][0] > start[0]]
                for other_start, other_end, other_task in active:
                    comparisons += 1
                    if comparisons > 1000000:
                        return False
                    if other_task != task_id and all(a < d and c < b for a, b, c, d in zip(start, end, other_start, other_end)):
                        return False
                active.append((start, end, task_id))
        return True
    except (ValueError, TypeError, ZeroDivisionError):
        return False
