"""Four-mode experiments must share input IR, including shapes and annotations."""

import pytest
from tilelang import tvm
from examples.ascend.layout_workloads.catalog import CASES, MODES, mode_configs
from examples.ascend.layout_workloads.kernels import make_kernel


@pytest.mark.parametrize("case", list(CASES))
def test_modes_have_identical_input_ir(case):
    native = make_kernel(case)
    for mode in MODES:
        with tvm.transform.PassContext(config=mode_configs(mode)):
            candidate = make_kernel(case)
        tvm.ir.assert_structural_equal(native, candidate)
