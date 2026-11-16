"""Correctness entry for the frontend-provided GQA pipeline schedule."""

import tilelang.testing

from example_gqa_manual_schedule import check_correctness


@tilelang.testing.requires_ascend
def test_gqa_manual_schedule():
    # Reuse the example's BF16 inputs, finite-output check and reference/error
    # threshold. S2=1024 covers the eight KV tiles required by stage 7.
    check_correctness()
