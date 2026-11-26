"""Collect raw NPU metrics separately from ordinary benchmark samples."""
import json
from pathlib import Path
import uuid


def capture(fn, directory, repeats=10):
    import torch
    import torch_npu

    directory = Path(directory) / uuid.uuid4().hex
    directory.mkdir(parents=True)
    # Resolve lazy compilation/initialization before entering the capture.
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    handler = profiler.tensorboard_trace_handler(str(directory), analyse_flag=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.NPU],
        on_trace_ready=handler,
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            l2_cache=False, data_simplification=False),
    ) as prof:
        for _ in range(repeats):
            fn()
        torch.npu.synchronize()
        prof.step()
    result = {"directory": str(directory), "repeats": repeats,
              "metric": "PipeUtilization", "files":
              [str(p.relative_to(directory)) for p in sorted(directory.rglob("*")) if p.is_file()],
              "interpretation": "Raw metrics; utilization is not additive phase duration. "
                                "Inspect kernel identity, VECTOR/MTE metrics and trace overlap."}
    (directory / "capture.json").write_text(json.dumps(result, indent=2))
    return result
