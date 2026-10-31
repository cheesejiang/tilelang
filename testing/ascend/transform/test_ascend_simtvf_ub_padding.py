"""Regular cross-row UB padding: selection, physical lowering and fallbacks."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, region, seq
from testing.ascend.transform.test_ascend_simtvf_ub_ownership import _executed_store_indices, _infer, _parallel, _vf


def _program(stride=1, dtype="float32", columns=32, boundary=None):
    shape = (32 * stride, columns)
    a = tirx.decl_buffer(shape, dtype, name="a", scope="shared.dyn", data_alignment=32)
    # Sparse half stores have unknown RMW cost. Use full-word stores to test
    # padding of half loads without pretending that RMW is modeled.
    store_dtype = "float32"
    b = tirx.decl_buffer(shape, store_dtype, name="b", scope="shared.dyn", data_alignment=32)
    loop = _parallel((32,), lambda i: tirx.BufferStore(b, tirx.Cast(store_dtype, a[i * stride, 0]), [i * stride, 0]))
    annotations = {}
    extra = []
    params = []
    buffers = [a, b]
    if boundary == "layout":
        annotations = {"layout_map": {a.data: tilelang.layout.Layout(shape, lambda i, j: (i, j))}}
    elif boundary == "conditional":
        flag = tirx.Var("flag", "int32")
        params = [flag]
        loop = tirx.IfThenElse(flag > 0, loop, None)
    elif boundary == "escape":
        extra = [tirx.Evaluate(tirx.call_extern("int32", "escape", a.data))]
    elif boundary == "dma":
        gm = tirx.decl_buffer(shape, dtype, name="gm")
        params = [gm]
        extra = [copy(gm, a)]
    elif boundary == "capacity":
        buffers.append(tirx.decl_buffer((216 * 1024 // 4,), "float32", name="reserved", scope="shared.dyn"))
    body = seq(*extra, _vf(loop, 32)) if extra else _vf(loop, 32)
    return kernel(body, buffers=buffers, params=params, annotations=annotations), a, b


def _storage_layouts(mod):
    result = {}
    for block in nodes(mod, tirx.SBlock):
        if "layout_map" in block.annotations:
            result.update({str(buffer.name): layout for buffer, layout in block.annotations["layout_map"].items()})
    return result


def _period_rows(layout, rows, columns, element_bytes):
    analyzer = tvm.arith.Analyzer()
    offsets = [int(analyzer.simplify(layout.map_forward_index([r, 0])[0])) - r * columns for r in range(rows)]
    period = next(r for r, offset in enumerate(offsets) if offset)
    for r, offset in enumerate(offsets):
        assert offset == (r // period) * (32 // element_bytes)
    return period


def _expected_period(stride, element_bytes, columns=32):
    # With fixed 32B holes, a 64B (mod 128B) lane step already spans two
    # group residues; a 0B (mod 128B) step spans just one. For 32 active lanes,
    # the longest equally cheap periods are respectively 16 and 8 lane steps.
    delta = stride * columns * element_bytes
    assert delta % 128 in (0, 64)
    return stride * (16 if delta % 128 == 64 else 8)


@pytest.mark.parametrize("stride", [1, 2, 16])
@pytest.mark.parametrize("dtype,element_bytes", [("float16", 2), ("float32", 4), ("int64", 8)])
def test_cross_row_padding_period_and_lowering(stride, dtype, element_bytes):
    mod, a, b = _program(stride, dtype)
    inferred = _infer(mod)
    layouts = _storage_layouts(inferred)
    analyzer = tvm.arith.Analyzer()
    for buffer in (a, b):
        buffer_bytes = element_bytes if buffer.same_as(a) else 4
        layout = layouts[str(buffer.name)]
        # Fixed 32B holes; choose the longest equally cheap regular period.
        period = _expected_period(stride, buffer_bytes)
        assert _period_rows(layout, 32 * stride, 32, buffer_bytes) == period
        for row in range(32 * stride):
            for col in (0, 31):
                address = layout.map_forward_index([row, col])[0]
                expected = row * 32 + col + (row // period) * (32 // buffer_bytes)
                assert analyzer.can_prove_equal(address, expected)
    with inferred["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(inferred)
    physical = {str(buffer.name): buffer for block in nodes(lowered, tirx.SBlock) for buffer in block.alloc_buffers}
    for name in ("a", "b"):
        assert len(physical[name].shape) == 1
        buffer_bytes = 4 if name == "b" else element_bytes
        period = _expected_period(stride, buffer_bytes)
        expected_elements = 32 * stride * 32 + ((32 * stride - 1) // period) * (32 // buffer_bytes)
        assert int(physical[name].shape[0]) == expected_elements
    # The pass must not replace the committed physical layout on a rerun.
    tvm.ir.assert_structural_equal(inferred, _infer(inferred))


@pytest.mark.parametrize("boundary", ["layout", "conditional", "escape", "capacity"])
def test_unsupported_padding_retains_storage(boundary):
    mod, _, _ = _program(boundary=boundary)
    layouts = _storage_layouts(_infer(mod))
    assert "b" not in layouts
    if boundary != "layout":
        assert "a" not in layouts


def test_already_distributed_cross_row_access_retains_storage():
    # Row size 17*8B already rotates through all groups.
    mod, _, _ = _program(columns=34)
    assert not _storage_layouts(_infer(mod))


def test_continuous_access_does_not_pad():
    a, b = [tirx.decl_buffer((32, 32), "float32", name=name, scope="shared.dyn", data_alignment=32) for name in "ab"]
    loop = _parallel((32, 32), lambda i, j: tirx.BufferStore(b, a[i, j], [i, j]))
    assert not _storage_layouts(_infer(kernel(_vf(loop, 128), buffers=[a, b])))


def test_padding_scores_every_access_to_the_buffer():
    # The frequent stride-1 access must outweigh the stride-16 hint encountered
    # first. Both loops use the same storage, but separate inference components.
    mod, a, b = _program(stride=16)
    crossing = nodes(mod, tirx.For)[0]
    frequent = _parallel((32,), lambda i: tirx.BufferStore(b, a[i, 0], [i, 0]))
    k = tirx.Var("k", "int32")
    repeated = tirx.For(k, 0, 32, tirx.ForKind.SERIAL, frequent)
    combined = kernel(_vf(seq(crossing, repeated), 32), buffers=[a, b])
    inferred = _infer(combined)
    layouts = _storage_layouts(inferred)
    assert set(layouts) == {"a", "b"}
    for layout in layouts.values():
        _period_rows(layout, 512, 32, 4)
    with inferred["main"].attrs["target"]:
        transform.AscendLowerTileOp()(inferred)


def test_unknown_partial_store_cost_retains_storage():
    a, b = [tirx.decl_buffer((32, 32), "float16", name=name, scope="shared.dyn", data_alignment=32) for name in "ab"]
    loop = _parallel((32,), lambda i: tirx.BufferStore(b, a[i, 0], [i, 0]))
    assert not _storage_layouts(_infer(kernel(_vf(loop, 32), buffers=[a, b])))


@pytest.mark.parametrize("stride", [1, 16])
def test_fixed_32b_padding_with_dma(stride):
    mod, a, b = _program(stride=stride)
    source, dest = [tirx.decl_buffer(a.shape, "float32", name=name) for name in ("source", "dest")]
    body = seq(copy(source, a), _vf(nodes(mod, tirx.For)[0], 32), copy(b, dest))
    inferred = _infer(kernel(body, buffers=[a, b], params=[source, dest]))
    layouts = _storage_layouts(inferred)
    assert set(layouts) == {"a", "b"}
    for layout in layouts.values():
        assert _period_rows(layout, 32 * stride, 32, 4) == 8 * stride
    with inferred["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(inferred)
    from testing.ascend._ir import calls

    for name in ("tl.ascend_copy_gm_to_ubuf", "tl.ascend_copy_ubuf_to_gm"):
        (dma,) = calls(lowered, name)
        analyzer = tvm.arith.Analyzer()
        assert analyzer.can_prove_equal(dma.args[3], 4)
        assert analyzer.can_prove_equal(dma.args[4], 8 * stride * 128)
        ub_pitch = dma.args[10] if name.endswith("gm_to_ubuf") else dma.args[7]
        assert analyzer.can_prove_equal(ub_pitch, 8 * stride * 128 + 32)


def test_partial_dma_input_retains_unpadded_storage():
    mod, a, b = _program()
    source = tirx.decl_buffer((16, 32), "float32", name="source")
    body = seq(copy(source, a, dst_shape=[16, 32]), _vf(nodes(mod, tirx.For)[0], 32))
    inferred = _infer(kernel(body, buffers=[a, b], params=[source]))
    assert "a" not in _storage_layouts(inferred)


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("stride", [1, 16])
def test_padding_through_full_tile_simt_copy(direction, stride):
    mod, a, b = _program(stride=stride)
    fragment = tirx.decl_buffer(a.shape, "float32", name="fragment", scope="local.fragment")
    transfer = copy(a, fragment) if direction == "load" else copy(fragment, a)
    crossing = nodes(mod, tirx.For)[0]
    k = tirx.Var("k", "int32")
    # Include the full copy in the score, but make the regular crossing access
    # frequent enough to justify padding despite any added dense-copy cost.
    repeated = tirx.For(k, 0, 32, tirx.ForKind.SERIAL, crossing)
    body = seq(repeated, transfer) if direction == "load" else seq(transfer, repeated)
    inferred = _infer(kernel(_vf(body, 32), buffers=[a, b, fragment]))
    layout = _storage_layouts(inferred)["a"]
    analyzer = tvm.arith.Analyzer()
    period = _period_rows(layout, 32 * stride, 32, 4)
    with inferred["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(inferred)
    accesses = [*nodes(lowered, tirx.BufferLoad), *nodes(lowered, tirx.BufferStore)]
    ub_accesses = [access for access in accesses if str(access.buffer.name) == "a"]
    # Both the compute and copy must use the padded physical buffer. No old
    # two-dimensional access may survive the copy lowering.
    assert len(ub_accesses) >= 2
    expected_size = 32 * stride * 32 + ((32 * stride - 1) // period) * 8
    for access in ub_accesses:
        assert len(access.indices) == 1
        assert int(access.buffer.shape[0]) == expected_size
    if direction == "store":
        # Execute the lowered loop coordinates on the host: the copy must
        # write every logical element once and never write the padding holes.
        expected = [(row * 32 + col + (row // period) * 8,) for row in range(32 * stride) for col in range(32)]
        assert sorted(_executed_store_indices(lowered, "a")) == expected


@pytest.mark.parametrize("kind", ["annotation", "escape"])
def test_unsupported_simt_copy_retains_unpadded_storage(kind):
    mod, a, b = _program()
    fragment = tirx.decl_buffer(a.shape, "float32", name="fragment", scope="local.fragment")
    kwargs = {}
    if kind == "annotation":
        kwargs = {"annotations": {"coalesced_width": tirx.IntImm("int32", 1)}}
    stmts = [copy(a, fragment, **kwargs), nodes(mod, tirx.For)[0]]
    if kind == "escape":
        stmts.append(tirx.Evaluate(tirx.call_extern("int32", "consume", a.data)))
    inferred = _infer(kernel(_vf(seq(*stmts), 32), buffers=[a, b, fragment]))
    layouts = _storage_layouts(inferred)
    assert "a" not in layouts
    assert "b" not in layouts


def _column_copy_program(iterations=32, sparse_replicas=False):
    work = tirx.decl_buffer((32, 32), "float32", name="work", scope="shared.dyn")
    values = tirx.decl_buffer((32, 1), "float32", name="values", scope="local.fragment")
    k = tirx.Var("column", "int32")
    transfer = copy(values, work, dst_indices=[0, k % 32], dst_shape=[32, 1])
    body = _vf(tirx.For(k, 0, iterations, tirx.ForKind.SERIAL, transfer), 32)
    # This is the immutable placement supplied by a preceding collective:
    # one result per lane, not a freely repartitionable whole matrix.
    placement = (tilelang.layout.Fragment(
        (32, 1), forward_thread_fn=lambda r, c, rep: (r % 2) * 16 + rep,
        forward_index_fn=lambda r, c: r // 2, replicate=16,
    ) if sparse_replicas else tilelang.layout.Fragment(
        (32, 1), forward_thread_fn=lambda r, c: r, forward_index_fn=lambda r, c: 0
    ))
    return kernel(body, buffers=[work, values], annotations={"layout_map": {values.data: placement}})


def test_partial_column_copy_padding_preserves_fragment_and_all_phases():
    program = _column_copy_program()
    with tvm.transform.PassContext(config={"tl.ascend_enable_ub_padding": False}):
        native = _infer(program)
    with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
        inferred = _infer(program)
    layouts = _storage_layouts(inferred)
    assert "work" not in _storage_layouts(native)
    period = _period_rows(layouts["work"], 32, 32, 4)
    assert period == 8
    after = [record for record in inferred["main"].attrs["tl.ascend_layout_diagnostics"]
             if str(record["stage"]) == "padding_after"]
    assert len(after) == 1
    assert int(after[0]["requests"]) == 32
    assert int(after[0]["entries"]) == 32 * 8
    assert str(after[0]["cost_scope"]) == "padding_scored_ops_only_invariant_ops_excluded"
    tvm.ir.assert_structural_equal(layouts["values"], _storage_layouts(native)["values"])
    with inferred["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(inferred)
    expected = [(r * 32 + c + (r // period) * 8,) for r in range(32) for c in range(32)]
    assert sorted(_executed_store_indices(lowered, "work")) == expected


def test_partial_copy_phase_budget_retains_native_storage():
    # The scorer must not extrapolate column-zero cost when it cannot cover
    # every invocation within its bounded phase enumeration budget.
    assert "work" not in _storage_layouts(_infer(_column_copy_program(1025)))


def test_padding_collective_results_with_nonadjacent_active_lanes():
    # Replica-zero stores are at lanes 0 and 16, with consecutive row results.
    # A hint collector that only inspects lane n/n+1 misses every row stride.
    with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
        inferred = _infer(_column_copy_program(sparse_replicas=True))
    layout = _storage_layouts(inferred)["work"]
    # Unlike a full 32-lane request, each request contains only TWO rows.
    # Every-row padding separates those two groups; every-eight-row padding
    # would leave almost all pairs in the same group.
    assert _period_rows(layout, 32, 32, 4) == 1
    records = list(inferred["main"].attrs["tl.ascend_layout_diagnostics"])
    assert all(int(r["padding_analysis_version"]) == 3 for r in records)
    assert any(str(r["stage"]) == "padding_stride_hint" and str(r["buffer"]) == "work" for r in records)
    before = next(r for r in records if str(r["stage"]) == "padding_before")
    after = next(r for r in records if str(r["stage"]) == "padding_after")
    assert int(before["requests"]) == int(after["requests"]) == 32 * 16
    assert int(before["entries"]) == 32 * 32
    assert int(after["entries"]) == 32 * 16


def test_padding_reports_buffer_rejection_before_candidate_generation():
    source = tirx.decl_buffer((1024,), "float32", name="source", scope="shared.dyn", data_alignment=32)
    dest = tirx.decl_buffer((32,), "float32", name="dest", scope="shared.dyn", data_alignment=32)
    loop = _parallel((32,), lambda i: tirx.BufferStore(dest, source[i * 32], [i]))
    program = kernel(_vf(loop, 32), buffers=[source, dest])
    with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
        inferred = _infer(program)
    records = inferred["main"].attrs["tl.ascend_layout_diagnostics"]
    assert any(str(r["stage"]) == "padding_buffer_rejected"
               and str(r["status"]) == "buffer_rank_less_than_two"
               and str(r["buffer"]) == "source" for r in records)


@pytest.mark.parametrize("boundary_name", ["a", "b"])
def test_padding_keeps_non_gm_dma_storage_unchanged(boundary_name):
    program, a, b = _program()
    boundary = a if boundary_name == "a" else b
    other = b if boundary_name == "a" else a
    l1 = tirx.decl_buffer((32, 32), "float32", name="l1", scope="shared.l1", data_alignment=32)
    body = seq(_vf(nodes(program, tirx.For)[0], 32), copy(boundary, l1))
    program = kernel(body, buffers=[a, b, l1])
    with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
        inferred = _infer(program)
    layouts = _storage_layouts(inferred)
    assert str(other.name) in layouts
    assert _period_rows(layouts[str(other.name)], 32, 32, 4) == 8
    if boundary_name in layouts:
        # A pre-existing identity layout is allowed; physical expansion is not.
        import math
        assert math.prod(int(x) for x in layouts[boundary_name].get_output_shape()) == 1024
    records = inferred["main"].attrs["tl.ascend_layout_diagnostics"]
    assert any(str(r["stage"]) == "padding_buffer_rejected"
               and str(r["status"]) == "unmodeled_operator_accesses_storage"
               and str(r["buffer"]) == boundary_name for r in records)


@pytest.mark.parametrize("reduce_from_output", [False, True])
def test_reducer_component_padding_only_changes_isolated_output(reduce_from_output):
    source = tirx.decl_buffer((32, 32), "float32", name="source", scope="shared.dyn")
    work = tirx.decl_buffer((32, 32), "float32", name="work", scope="shared.dyn")
    acc = tirx.decl_buffer((32, 1), "float32", name="acc", scope="local.reducer")
    values = tirx.decl_buffer((32, 1), "float32", name="values", scope="local.fragment")

    def tileop(name, *args):
        return tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get(name), list(args)))

    operand = work if reduce_from_output else source
    update = _parallel(
        (32, 32),
        lambda r, c: tileop("tl.reducer_update", acc[r, 0], operand[r, c]),
    )
    body = _vf(seq(
        tileop("tl.tileop.reducer_init", region(acc, 2)),
        update,
        tileop("tl.tileop.finalize_reducer_v2", region(acc, 3), region(values, 2)),
        copy(values, work, dst_shape=[32, 1]),
    ), 32)
    placement = tilelang.layout.Fragment(
        (32, 1), forward_thread_fn=lambda r, c: r, forward_index_fn=lambda r, c: 0
    )
    program = kernel(body, buffers=[source, work, acc, values], annotations={
        "reducer_info_v2": {acc.data: {"op": "sum"}},
        "layout_map": {values.data: placement},
    })
    with tvm.transform.PassContext(config={"tl.ascend_enable_ub_padding": False}):
        native = _infer(program)
    inferred = _infer(program)
    native_layouts, layouts = _storage_layouts(native), _storage_layouts(inferred)
    assert "work" not in native_layouts
    assert ("work" in layouts) == (not reduce_from_output)
    assert "source" not in layouts
    tvm.ir.assert_structural_equal(layouts["values"], native_layouts["values"])
    # The reducer update's ownership must remain exactly the native plan.
    def loop_layouts(mod):
        return [n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For)
                if "parallel_loop_layout" in n.annotations]
    for before, after in zip(loop_layouts(native), loop_layouts(inferred), strict=True):
        tvm.ir.assert_structural_equal(before, after)


@pytest.mark.parametrize("stride", [1, 2, 16])
def test_64_byte_rows_use_longer_period(stride):
    mod, _, _ = _program(stride=stride, columns=16)
    layouts = _storage_layouts(_infer(mod))
    for layout in layouts.values():
        expected = (16 if stride == 1 else 8) * stride
        assert _period_rows(layout, 32 * stride, 16, 4) == expected


def test_four_groups_do_not_benefit_from_32b_padding():
    # 160B row pitch already uses all four groups available in this residue
    # class. Padding by 32B cannot change the lower two group bits.
    mod, _, _ = _program(columns=40)
    assert not _storage_layouts(_infer(mod))


def test_layout_diagnostics_are_opt_in_and_preserve_decisions():
    mod, _, _ = _program()
    native = _infer(mod)
    assert "tl.ascend_layout_diagnostics" not in native["main"].attrs
    with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
        diagnosed = _infer(mod)
    tvm.ir.assert_structural_equal(native["main"].body, diagnosed["main"].body)
    records = list(diagnosed["main"].attrs["tl.ascend_layout_diagnostics"])
    final = next(r for r in records if str(r["stage"]) == "final")
    assert int(final["ub_extra_storage_bytes"]) == 2 * 96
    assert int(final["estimated_ub_extra_allocation_bytes"]) == 2 * 96
    assert str(final["allocation_estimate_method"]) == "sum_round_up_32B_without_alias_reuse"
    # Final collection runs after extracting the loop annotations: that step
    # must leave infer_list_ intact for the cost model.
    assert int(final["cost_known"]) == 1
    assert float(final["estimated_latency_cycles"]) > 0
    assert float(final["service_cycles"]) > 0
    assert int(final["instructions"]) > 0
    assert int(final["requests"]) > 0
    assert int(final["entries"]) > 0
    assert str(final["dma_cost"]) == "not_modeled"
    assert str(final["cost_scope"]) == "function_simt_ub_only"
    assert {str(b["buffer"]) for b in final["buffers"]} == {"a", "b"}
    assert all("Layout" in str(b["layout"]) for b in final["buffers"])
    assert all(int(b["padding_bytes"]) == 32 for b in final["buffers"])
    assert any(str(r["stage"]) == "padding_candidate" for r in records)


def test_padding_capacity_does_not_charge_512b_per_buffer():
    mod, a, b = _program()
    # Two 4096B buffers become 4192B each. Leave exactly that much room
    # under the pass budget; artificial 512B rounding would reject padding.
    budget = 216 * 1024
    reserved = tirx.decl_buffer(((budget - 2 * 4192) // 4,), "float32",
                               name="reserved", scope="shared.dyn", data_alignment=32)
    combined = kernel(_vf(nodes(mod, tirx.For)[0], 32), buffers=[a, b, reserved])
    with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
        inferred = _infer(combined)
    assert set(_storage_layouts(inferred)) == {"a", "b"}
    final = next(r for r in inferred["main"].attrs["tl.ascend_layout_diagnostics"]
                 if str(r["stage"]) == "final")
    assert int(final["estimated_ub_allocation_bytes"]) == budget
    assert int(final["estimated_ub_extra_allocation_bytes"]) == 192


@pytest.mark.parametrize("ownership", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_independent_ub_optimization_switches(ownership, padding):
    mod, _, _ = _program()
    config = {"tl.ascend_enable_ub_ownership": ownership,
              "tl.ascend_enable_ub_padding": padding,
              "tl.ascend_layout_diagnostics": True}
    with tvm.transform.PassContext(config=config):
        inferred = _infer(mod)
    assert bool(_storage_layouts(inferred)) == padding
    final = next(r for r in inferred["main"].attrs["tl.ascend_layout_diagnostics"] if str(r["stage"]) == "final")
    assert int(final["ownership_enabled"]) == int(ownership)
    assert int(final["padding_enabled"]) == int(padding)
    if ownership and padding:
        tvm.ir.assert_structural_equal(inferred["main"].body, _infer(mod)["main"].body)
    if padding and not ownership:
        with tvm.transform.PassContext(config={"tl.ascend_enable_ub_ownership": False,
                                              "tl.ascend_enable_ub_padding": False}):
            baseline = _infer(mod)
        def loop_layouts(module):
            return [loop.annotations["parallel_loop_layout"] for loop in nodes(module, tirx.For)
                    if "parallel_loop_layout" in loop.annotations]
        before, after = loop_layouts(baseline), loop_layouts(inferred)
        assert len(before) == len(after)
        for a, b in zip(before, after):
            tvm.ir.assert_structural_equal(a, b)


@pytest.mark.parametrize("enabled", [False, True])
def test_ownership_switch_controls_contiguous_search(enabled):
    from testing.ascend.transform.test_ascend_simtvf_ub_ownership import _elementwise

    mod = _elementwise(n=8192, threads=128, alignments=(32, 32, 32))
    with tvm.transform.PassContext(config={"tl.ascend_enable_ub_ownership": enabled,
                                          "tl.ascend_enable_ub_padding": False,
                                          "tl.ascend_layout_diagnostics": True}):
        inferred = _infer(mod)
    records = inferred["main"].attrs["tl.ascend_layout_diagnostics"]
    assert any(str(r["stage"]) == "ownership_before" for r in records) == enabled
    assert not _storage_layouts(inferred)


def test_two_dimensional_cross_row_ownership_reaches_candidate_search():
    a, b = [tirx.decl_buffer((32, 32), "float32", name=name,
                             scope="shared.dyn", data_alignment=32) for name in "ab"]
    loop = _parallel((32, 32), lambda c, r: tirx.BufferStore(b, a[r, c], [r, c]))
    mod = kernel(_vf(loop, 32), buffers=[a, b])
    with tvm.transform.PassContext(config={"tl.ascend_enable_ub_ownership": True,
                                          "tl.ascend_enable_ub_padding": False,
                                          "tl.ascend_layout_diagnostics": True}):
        inferred = _infer(mod)
    records = list(inferred["main"].attrs["tl.ascend_layout_diagnostics"])
    before = [r for r in records if str(r["stage"]) == "ownership_before"]
    after = [r for r in records if str(r["stage"]) == "ownership_after"]
    assert before and len(before) == len(after)
    for initial, selected in zip(before, after):
        assert int(initial["component"]) == int(selected["component"])
        assert float(selected["estimated_latency_cycles"]) <= float(initial["estimated_latency_cycles"])
    assert not _storage_layouts(inferred)
