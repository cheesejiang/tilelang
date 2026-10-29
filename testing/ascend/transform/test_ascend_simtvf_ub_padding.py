"""Regular cross-row UB padding: selection, physical lowering and fallbacks."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, seq
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
        # K=stride*columns; exactly one 8B hole between adjacent lanes.
        for row in range(32 * stride):
            for col in (0, 31):
                address = layout.map_forward_index([row, col])[0]
                expected = row * 32 + col + (row // stride) * (8 // buffer_bytes)
                assert analyzer.can_prove_equal(address, expected)
    with inferred["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(inferred)
    physical = {str(buffer.name): buffer for block in nodes(lowered, tirx.SBlock) for buffer in block.alloc_buffers}
    for name in ("a", "b"):
        assert len(physical[name].shape) == 1
        buffer_bytes = 4 if name == "b" else element_bytes
        expected_elements = 32 * stride * 32 + 31 * (8 // buffer_bytes)
        assert int(physical[name].shape[0]) == expected_elements
    # The pass must not replace the committed physical layout on a rerun.
    tvm.ir.assert_structural_equal(inferred, _infer(inferred))


@pytest.mark.parametrize("boundary", ["layout", "conditional", "escape", "dma", "capacity"])
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
        assert tvm.arith.Analyzer().can_prove_equal(layout.map_forward_index([16, 0])[0], 16 * 34)
    with inferred["main"].attrs["target"]:
        transform.AscendLowerTileOp()(inferred)


def test_unknown_partial_store_cost_retains_storage():
    a, b = [tirx.decl_buffer((32, 32), "float16", name=name, scope="shared.dyn", data_alignment=32) for name in "ab"]
    loop = _parallel((32,), lambda i: tirx.BufferStore(b, a[i, 0], [i, 0]))
    assert not _storage_layouts(_infer(kernel(_vf(loop, 32), buffers=[a, b])))


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
    for row in range(32 * stride):
        assert analyzer.can_prove_equal(layout.map_forward_index([row, 0])[0], row * 32 + (row // stride) * 2)
    with inferred["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(inferred)
    accesses = [*nodes(lowered, tirx.BufferLoad), *nodes(lowered, tirx.BufferStore)]
    ub_accesses = [access for access in accesses if str(access.buffer.name) == "a"]
    # Both the compute and copy must use the padded physical buffer. No old
    # two-dimensional access may survive the copy lowering.
    assert len(ub_accesses) >= 2
    expected_size = 32 * stride * 32 + 31 * 2
    for access in ub_accesses:
        assert len(access.indices) == 1
        assert int(access.buffer.shape[0]) == expected_size
    if direction == "store":
        # Execute the lowered loop coordinates on the host: the copy must
        # write every logical element once and never write the padding holes.
        expected = [(row * 32 + col + (row // stride) * 2,) for row in range(32 * stride) for col in range(32)]
        assert sorted(_executed_store_indices(lowered, "a")) == expected


@pytest.mark.parametrize("kind", ["partial", "annotation", "escape"])
def test_unsupported_simt_copy_retains_unpadded_storage(kind):
    mod, a, b = _program()
    fragment = tirx.decl_buffer(a.shape, "float32", name="fragment", scope="local.fragment")
    kwargs = {}
    if kind == "partial":
        kwargs = {"src_shape": [16, 32], "dst_shape": [16, 32]}
    elif kind == "annotation":
        kwargs = {"annotations": {"coalesced_width": tirx.IntImm("int32", 1)}}
    stmts = [copy(a, fragment, **kwargs), nodes(mod, tirx.For)[0]]
    if kind == "escape":
        stmts.append(tirx.Evaluate(tirx.call_extern("int32", "consume", fragment.data)))
    inferred = _infer(kernel(_vf(seq(*stmts), 32), buffers=[a, b, fragment]))
    layouts = _storage_layouts(inferred)
    assert "a" not in layouts
    assert "b" not in layouts
