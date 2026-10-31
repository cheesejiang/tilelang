"""Check lane/mask evidence independently of inferred ownership choices."""

import pytest
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import kernel
from testing.ascend.transform.test_ascend_simtvf_ub_ownership import _vf
from examples.ascend.layout_workloads.column_store_audit import collect_column_store_audit


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("active", [1, 32])
def test_column_store_audit_uses_actual_addresses_and_mask(padded, active):
    tx = tirx.IterVar(tvm.ir.Range(0, 32), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    buffer = tirx.decl_buffer((1048 if padded else 1024,), "float32", name="scores_out", scope="shared.dyn", data_alignment=32)
    offset = tx.var * 32 + ((tx.var // 8) * 8 if padded else 0)
    store = tirx.BufferStore(buffer, tirx.FloatImm("float32", 1), [offset])
    mod = kernel(_vf(tirx.IfThenElse(tx.var < active, store, None), tx=tx, threads=32), buffers=[buffer])
    report = collect_column_store_audit(mod, "scores_out")
    assert report["status"] == "known"
    assert report["max_active_lanes"] == active
    assert report["conflicting_batches"] == int(active > 1)
    assert report["max_distinct_lines_per_group"] == (1 if active == 1 else 8 if padded else 32)


def test_column_store_audit_does_not_guess_unknown_predicate():
    tx = tirx.IterVar(tvm.ir.Range(0, 32), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    flag = tirx.Var("flag", "int32")
    buffer = tirx.decl_buffer((1024,), "float32", name="scores_out", scope="shared.dyn", data_alignment=32)
    store = tirx.BufferStore(buffer, tirx.FloatImm("float32", 1), [tx.var * 32])
    mod = kernel(_vf(tirx.IfThenElse(flag > 0, store, None), tx=tx, threads=32), buffers=[buffer], params=[flag])
    assert collect_column_store_audit(mod, "scores_out")["status"] == "unknown"


@pytest.mark.parametrize("extent", [1, 2])
@pytest.mark.parametrize("axis", ["threadIdx.y", "threadIdx.z"])
def test_column_store_audit_accepts_only_singleton_secondary_axes(axis, extent):
    tx = tirx.IterVar(tvm.ir.Range(0, 32), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    secondary = tirx.IterVar(tvm.ir.Range(0, extent), tirx.Var("secondary", "int32"), tirx.IterVar.ThreadIndex, axis)
    buffer = tirx.decl_buffer((1024,), "float32", name="scores_out", scope="shared.dyn", data_alignment=32)
    store = tirx.BufferStore(buffer, tirx.FloatImm("float32", 1), [tx.var * 32])
    body = tirx.AttrStmt(secondary, "thread_extent", extent, _vf(store, tx=tx, threads=32))
    report = collect_column_store_audit(kernel(body, buffers=[buffer]), "scores_out")
    assert report["status"] == ("known" if extent == 1 else "unknown")
