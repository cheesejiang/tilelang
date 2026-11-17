"""Worker rewriting must preserve task coverage and skip dependent schedules."""
import importlib.util
from pathlib import Path

import pytest
from tilelang import tvm
from tvm import tirx
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.block_policy import bind_block_policy, resolve_core_counts
from tilelang.ascend.transform import AutoBlocks
from testing.ascend._ir import nodes


def example(name):
    path = Path(__file__).resolve().parents[3] / "examples" / "ascend" / f"example_{name}.py"
    spec = importlib.util.spec_from_file_location(f"auto_blocks_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize(func, aic=28, aiv=56, enabled=True):
    func = func.with_attr("tl.ascend.auto_blocks", int(enabled)).with_attr("tl.ascend.aic_cores", aic).with_attr("tl.ascend.aiv_cores", aiv)
    mod = tvm.IRModule({"main": func})
    return tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True, lower_thread_binding=False, default_threads=None,
        unsupported_annotations=["cluster_dims"], launch_dim_tags=["cthread"],
    )(mod)


def grids(mod):
    return [int(node.value) for node in nodes(mod, tirx.AttrStmt)
            if node.attr_key == "thread_extent" and getattr(node.node, "thread_tag", "") == "blockIdx.x"]


def test_budget_priority_and_unavailable_fallback():
    manual = {"TILELANG_ASCEND_AIC_CORES": "14", "TILELANG_ASCEND_AIV_CORES": "32"}
    assert resolve_core_counts(manual, lambda: (28, 56)) == (28, 56)
    assert resolve_core_counts(manual, lambda: (None, None)) == (14, 32)
    assert resolve_core_counts({}, lambda: (None, None)) == (0, 0)
    assert resolve_core_counts(manual, lambda: (28, None)) == (28, 32)
    with pytest.raises(ValueError):
        resolve_core_counts({"TILELANG_ASCEND_AIC_CORES": "0"}, lambda: (None, None))


def test_compiler_report_records_actual_rewrite(monkeypatch, tmp_path):
    import json
    path = tmp_path / "compiler-blocks.jsonl"
    monkeypatch.setenv("TILELANG_ASCEND_AUTO_BLOCKS_REPORT", str(path))
    AutoBlocks()(materialize(example("vecadd").vector_add(2**20)))
    record = json.loads(path.read_text())
    assert record["enabled"] is True
    assert record["changed"] is True
    assert record["before"] == [64]
    assert record["after"] == [56]


@pytest.mark.parametrize("mode", ["simt", "simd"])
@pytest.mark.parametrize("cores", [28, 56, 57, 200])
def test_vector_coverage_with_new_tail(mode, cores):
    before = materialize(example("vecadd").vector_add(2**20, mode), aiv=cores)
    after = AutoBlocks()(before)
    blocks = min(cores, 128)
    assert grids(after) == [blocks]
    # Inspect the actual generated task guard/index, not just the launch grid.
    func = after["main"]
    grid = next(node for node in nodes(func, tirx.AttrStmt)
                if node.attr_key == "thread_extent" and getattr(node.node, "thread_tag", "") == "blockIdx.x")
    loop = next(node for node in nodes(grid.body, tirx.For) if "num_stages" in node.annotations)
    assert int(loop.extent) == (128 + blocks - 1) // blocks
    write = next(call for call in nodes(loop.body, tirx.Call)
                 if str(getattr(call.op, "name", "")) == "tl.region"
                 and int(call.args[1]) == 2 and call.args[0].buffer.scope() == "global")
    analyzer = tvm.arith.Analyzer()
    offsets = []
    for step in range(int(loop.extent)):
        for bx in range(blocks):
            mapping = {loop.loop_var: tirx.IntImm("int32", step), grid.node.var: tirx.IntImm("int32", bx)}
            valid = True
            if isinstance(loop.body, tirx.IfThenElse):
                valid = bool(int(analyzer.simplify(tirx.stmt_functor.substitute(loop.body.condition, mapping))))
            if valid:
                offsets.append(int(analyzer.simplify(tirx.stmt_functor.substitute(write.args[0].indices[0], mapping))))
    assert sorted(offsets) == list(range(0, 2**20, 8192))


@pytest.mark.parametrize("size,expected", [(1024, 16), (8192, 28)])
@pytest.mark.parametrize("dtype,mixed", [("bfloat16", False), ("float32", False), ("float32", True)])
def test_persistent_gemm_uses_cube_budget(size, expected, dtype, mixed):
    func = example("gemm").gemm(size, size, size, dtype=dtype, out_dtype=dtype, MIXED=mixed)
    assert grids(AutoBlocks()(materialize(func))) == [expected]


def test_no_budget_and_disabled_preserve_original():
    func = example("vecadd").vector_add(2**20)
    for before in (materialize(func, aic=0, aiv=0), materialize(func, enabled=False)):
        assert tvm.ir.structural_equal(AutoBlocks()(before), before)


def test_fixed_contiguous_schedule_is_not_rewritten():
    @T.prim_func
    def contiguous(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(32) as bx:
            a = T.alloc_shared((4,), "float32")
            for step in T.serial(1):
                begin = (bx + step) * 4
                T.copy(A[begin:begin + 4], a)
                with T.SimtVF(threads=4):
                    for i in T.Parallel(4):
                        a[i] = a[i] + 1
                T.copy(a, B[begin:begin + 4])
    before = materialize(contiguous)
    assert tvm.ir.structural_equal(AutoBlocks()(before), before)


def test_bound_device_counts_participate_in_cache_identity(monkeypatch):
    func = example("vecadd").vector_add(2**20)
    from tilelang.ascend import block_policy
    monkeypatch.setattr(block_policy, "resolve_core_counts", lambda: (28, 56))
    first = bind_block_policy(func)
    monkeypatch.setattr(block_policy, "resolve_core_counts", lambda: (14, 28))
    second = bind_block_policy(func)
    assert not tvm.ir.structural_equal(first, second)


def test_jit_memory_cache_refreshes_with_device_budget(monkeypatch):
    from tilelang.ascend import block_policy
    budget = [28, 56]
    monkeypatch.setattr(block_policy, "resolve_core_counts", lambda: tuple(budget))
    monkeypatch.delenv("TILELANG_ASCEND_AUTO_BLOCKS", raising=False)

    @tilelang.jit(target="ascend")
    def factory(n: int):
        @T.prim_func
        def kernel():
            T.evaluate(0)
        return kernel

    compiled = []
    def compile_once(*args, **kwargs):
        kernel = object()
        compiled.append(kernel)
        return kernel
    monkeypatch.setattr(factory, "compile", compile_once)
    first = factory(1)
    assert factory(1) is first
    budget[:] = [14, 28]
    assert factory(1) is not first
    assert len(compiled) == 2


def test_feedback_buffer_is_not_rewritten():
    @T.prim_func
    def feedback(A: T.Tensor((128,), "float32")):
        with T.Kernel(8) as bx:
            a = T.alloc_shared((4,), "float32")
            for step in T.serial(4):
                begin = (step * 8 + bx) * 4
                T.copy(A[begin:begin + 4], a)
                with T.SimtVF(threads=4):
                    for i in T.Parallel(4):
                        a[i] = a[i] + 1
                T.copy(a, A[begin:begin + 4])
    before = materialize(feedback, aiv=6)
    assert tvm.ir.structural_equal(AutoBlocks()(before), before)


def test_cross_task_scratch_state_is_not_rewritten():
    @T.prim_func
    def recurrent(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(8) as bx:
            a = T.alloc_shared((4,), "float32")
            carry = T.alloc_shared((4,), "float32")
            for step in T.serial(4):
                task = step * 8 + bx
                if task < 8:
                    T.copy(A[0:4], carry)
                T.copy(A[task * 4:task * 4 + 4], a)
                with T.SimtVF(threads=4):
                    for i in T.Parallel(4):
                        carry[i] = carry[i] + a[i]
                T.copy(carry, B[task * 4:task * 4 + 4])
    before = materialize(recurrent, aiv=6)
    assert tvm.ir.structural_equal(AutoBlocks()(before), before)
