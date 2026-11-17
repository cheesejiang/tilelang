"""Resolve compile-time Ascend core budgets without requiring an NPU runtime."""
from __future__ import annotations

import os


def _positive(value, name):
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if str(result) != str(value) or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def query_current_core_counts():
    """Return the selected device's Cube/AIV counts, or None when unavailable."""
    try:
        import torch_npu

        npu = torch_npu.npu
        if not npu.is_available():
            return None, None
        props = npu.get_device_properties(npu.current_device())
        return (getattr(props, "cube_core_num", None), getattr(props, "vector_core_num", None))
    except (ImportError, RuntimeError, AttributeError, OSError):
        return None, None


def resolve_core_counts(environ=None, query=None):
    """Device first, manual fallback per resource; zero denotes no budget."""
    environ = os.environ if environ is None else environ
    manual = [_positive(environ.get(f"TILELANG_ASCEND_{unit}_CORES"), unit) for unit in ("AIC", "AIV")]
    device = (query or query_current_core_counts)()
    result = []
    for detected, fallback in zip(device, manual):
        try:
            detected = _positive(detected, "device core count")
        except ValueError:
            detected = None
        result.append(detected or fallback or 0)
    return tuple(result)


def bind_block_policy(func):
    """Freeze budgets in the input PrimFunc, making them part of cache identity."""
    attrs = func.attrs or {}
    if "tl.ascend.auto_blocks" in attrs:
        return func
    enabled = os.environ.get("TILELANG_ASCEND_AUTO_BLOCKS", "1").lower() not in ("0", "false", "off")
    func = func.with_attr("tl.ascend.auto_blocks", int(enabled))
    if enabled:
        aic, aiv = resolve_core_counts()
        func = func.with_attr("tl.ascend.aic_cores", aic).with_attr("tl.ascend.aiv_cores", aiv)
    return func
