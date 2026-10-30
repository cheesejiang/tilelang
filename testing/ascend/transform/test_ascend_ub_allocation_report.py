"""Report allocated UB extents, including allocator reuse, without re-estimation."""

import pytest
from tilelang import tvm
from tvm import tirx
from tilelang.ascend import transform

from examples.ascend.ub_layout_ablation.layout_diagnostics import collect_final_ub_allocations, final_ir_fields


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_report_reads_merged_ub_allocation(disable_reuse):
    a = tirx.decl_buffer((1024,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((1048,), "float32", name="b", scope="shared.dyn")
    l1 = tirx.decl_buffer((4096,), "float32", name="l1", scope="shared.l1")
    body = tirx.SeqStmt([tirx.AllocBuffer(a), tirx.AllocBuffer(b), tirx.AllocBuffer(l1),
                         tirx.BufferStore(a, tirx.const(1, "float32"), [0]),
                         tirx.BufferStore(b, tirx.const(2, "float32"), [0])])
    thread = tirx.IterVar(tvm.ir.Range(0, 1), tirx.Var("block", "int32"),
                         tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.AttrStmt(thread, "thread_extent", 1, body)
    body = tirx.AttrStmt({a.data: [b.data], b.data: [a.data]}, "tl.buffer_alias_map", 1, body)
    mod = tvm.IRModule({"main": tirx.PrimFunc([], body)})
    merged = transform.MergeUBAllocations(align_bytes=32, disable_reuse=disable_reuse)(mod)
    report = collect_final_ub_allocations(merged)
    expected = 4096 + 4192 if disable_reuse else 4192
    assert report["status"] == "known"
    assert report["ub_allocation_bytes"] == expected
    (function,) = report["functions"]
    (allocation,) = function["allocations"]
    assert allocation["bytes"] == expected
    assert allocation["range_relative_to_buffer_base"] == [0, expected]


def test_report_uses_maximum_across_entry_functions():
    functions = {}
    for name, size in [("first", 100), ("second", 200)]:
        buffer = tirx.decl_buffer((size,), "uint8", name=name, scope="shared.dyn")
        functions[name] = tirx.PrimFunc([], tirx.AllocBuffer(buffer))
    report = collect_final_ub_allocations(tvm.IRModule(functions))
    assert report["ub_allocation_bytes"] == 200  # Do not sum unrelated entries.


def test_dynamic_extent_remains_unknown():
    n = tirx.Var("n", "int32")
    buffer = tirx.decl_buffer((n,), "float32", name="dynamic", scope="shared.dyn")
    report = collect_final_ub_allocations(tvm.IRModule({"main": tirx.PrimFunc([n], tirx.AllocBuffer(buffer))}))
    assert report["status"] == "dynamic_extent"
    assert report["ub_allocation_bytes"] is None
    assert final_ir_fields({"final_ir_ub": report})["final_ir_ub_allocation_bytes"] == "unknown"


def test_baseline_delta_requires_a_captured_baseline():
    report = {"final_ir_ub": {"status": "known", "ub_allocation_bytes": 4320}}
    baseline = {"final_ir_ub": {"status": "known", "ub_allocation_bytes": 4224}}
    assert final_ir_fields(report, baseline)["final_ir_ub_extra_vs_baseline_bytes"] == 96
    assert final_ir_fields(report)["final_ir_ub_extra_vs_baseline_bytes"] == "unknown"
    assert final_ir_fields({})["final_ir_ub_status"] == "missing_snapshot"
