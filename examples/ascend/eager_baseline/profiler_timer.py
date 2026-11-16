"""TileOPs-style cold-cache device-busy timing; no event fallback."""
import contextlib
from pathlib import Path

from profile_parser import _parse_ffts_profile

PROTOCOL = "ascend-profiler-cold-cache-v1"
FLUSH_BYTES = 1 << 30
FLUSH_NAME = "_tilelang_profiler_cache_flush"


@contextlib.contextmanager
def profile(directory):
    import torch_npu
    directory.mkdir(parents=True, exist_ok=False)
    handler = torch_npu.profiler.tensorboard_trace_handler(str(directory.resolve()), analyse_flag=False)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=handler,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False, data_simplification=False),
    ) as prof:
        yield prof


class ProfilerTimer:
    def __init__(self, directory, torch):
        from tilelang.profiler.msprof import get_msprof_cache_flush_kernel
        self.directory = Path(directory)
        self.torch = torch
        self.buffer = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="npu")
        self.flush = get_msprof_cache_flush_kernel(FLUSH_BYTES // 4, "int32")
        # Resolve and validate the flush name on this installed CANN version.
        with profile(self.directory / "flush-check") as prof:
            self.flush(self.buffer)
            torch.npu.synchronize()
            prof.step()
        _parse_ffts_profile(Path(prof.prof_if.prof_path), 1,
                            exclude_name_marker=FLUSH_NAME, require_marker_seen=True)

    def prepare(self):
        self.flush(self.buffer)
        self.torch.npu.synchronize()

    def warmup(self, fn):
        self.prepare()
        output = fn()
        self.torch.npu.synchronize()
        del output

    def measure(self, fn, name, index):
        self.prepare()
        with profile(self.directory / name / f"repeat-{index:02d}") as prof:
            output = fn()
            self.torch.npu.synchronize()
            prof.step()
        del output
        record = _parse_ffts_profile(Path(prof.prof_if.prof_path), 1, exclude_name_marker=FLUSH_NAME)
        if record["profile_excluded_kernel_count"] or record["profile_kernel_count"] <= 0 or record["latency_ms"] <= 0:
            raise RuntimeError("Profiler window contains cache flush or no positive device kernel activity")
        return record
