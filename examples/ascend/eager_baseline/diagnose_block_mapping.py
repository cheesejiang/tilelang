"""Map logical blocks to raw physical core IDs in an instrumented Cube GEMM.

Diagnostic only: scalar recording and the final device fence perturb execution.
No profiler or eager baseline is nested. The original GEMM DSL is unchanged.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

RECORD_WORDS = 16  # 128 bytes per block: avoid inter-block cache-line sharing.
MAGIC = 0xB10C3510
COLUMNS = ("magic", "logical_block", "physical_core_raw", "start_cycles",
           "end_cycles", "issued_output_tiles", "physical_core_exit_raw")


def instrument_source(code, output_elements):
    """Instrument the one pure-Cube BF16 kernel without changing its launch ABI.

    The caller allocates a record tail behind the output tensor's backing storage.
    Records are addressed from C + output_elements; C's tensor view stays M x N.
    """
    marker = "// TILELANG_BLOCK_MAPPING_DIAGNOSTIC"
    if marker in code:
        return code
    functions = list(re.finditer(r'extern\s+"C"\s+__global__\s+__cube__\s+void\s+\w+\s*\([^)]*\)\s*\{', code))
    if len(functions) != 1:
        raise ValueError("Expected exactly one pure Cube kernel")
    match = functions[0]
    if not re.search(r'__gm__\s+bfloat16_t\s*\*\s*C\b', match.group()):
        raise ValueError("Expected BF16 output pointer named C")
    opening = match.end() - 1
    depth, closing = 1, None
    for index in range(opening + 1, len(code)):
        if code[index] == '{':
            depth += 1
        elif code[index] == '}':
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing is None:
        raise ValueError("Unbalanced kernel body")
    body = code[opening + 1:closing]
    if body.count("asc_init();") != 1:
        raise ValueError("Expected exactly one asc_init")
    # Increment only on the actual guarded output-tile writeback path.
    store = re.compile(r'(?m)^(\s*)asc_copy_l0c2gm\(')
    if len(store.findall(body)) != 1:
        raise ValueError("Expected one output tile-store site in the original Cube GEMM")
    body = store.sub(lambda m: m.group(1) + '++__tl_diag_tiles;\n' + m.group(0), body)
    begin = '''
  // TILELANG_BLOCK_MAPPING_DIAGNOSTIC
  const int64_t __tl_diag_start = asc_get_system_cycle();
  const int64_t __tl_diag_core = asc_get_core_id();
  int64_t __tl_diag_tiles = 0;
'''
    body = body.replace('asc_init();', 'asc_init();' + begin)
    end = f'''
  // Wait for compute/writeback before taking the completion timestamp.
  asc_sync();
  const int64_t __tl_diag_end = asc_get_system_cycle();
  const int64_t __tl_diag_exit_core = asc_get_core_id();
  const int64_t __tl_diag_block = static_cast<int64_t>(block_idx);
  volatile __gm__ int64_t* __tl_diag_row =
      reinterpret_cast<volatile __gm__ int64_t*>(C + {output_elements})
      + __tl_diag_block * {RECORD_WORDS};
  __tl_diag_row[1] = __tl_diag_block;
  __tl_diag_row[2] = __tl_diag_core;
  __tl_diag_row[3] = __tl_diag_start;
  __tl_diag_row[4] = __tl_diag_end;
  __tl_diag_row[5] = __tl_diag_tiles;
  __tl_diag_row[6] = __tl_diag_exit_core;
  __tl_diag_row[0] = {MAGIC};
  // Publish scalar GM records before the kernel returns.
  asc_dcci_entire_out();
  asc_sync();
'''
    return code[:opening + 1] + body + end + code[closing:]


def decode_records(records, blocks, tasks, ticks_per_us):
    if len(records) != blocks:
        raise ValueError("Incomplete record count")
    result = []
    for block, values in enumerate(records):
        if len(values) < len(COLUMNS):
            raise ValueError("Truncated record")
        row = dict(zip(COLUMNS, map(int, values)))
        expected = len(range(block, tasks, blocks))
        if row['magic'] != MAGIC or row['logical_block'] != block:
            raise ValueError(f"Block {block}: missing/invalid record; check that instrumented code ran")
        if row['issued_output_tiles'] != expected:
            raise ValueError(f"Block {block}: tile count {row['issued_output_tiles']} != {expected}")
        if row['end_cycles'] < row['start_cycles']:
            raise ValueError(f"Block {block}: invalid timestamp order")
        row.update(blocks=blocks, expected_tiles=expected,
                   core_changed=row['physical_core_raw'] != row['physical_core_exit_raw'])
        result.append(row)
    origin = min(row['start_cycles'] for row in result)
    for row in result:
        row.update(start_us=(row['start_cycles'] - origin) / ticks_per_us,
                   end_us=(row['end_cycles'] - origin) / ticks_per_us,
                   duration_us=(row['end_cycles'] - row['start_cycles']) / ticks_per_us)
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(args):
    os.environ['TILELANG_DEFAULT_TARGET'] = 'ascend'
    os.environ['TILELANG_EXECUTION_BACKEND'] = 'tvm_ffi'
    os.environ['ASCEND_NPU_ARCH'] = 'dav-3510'
    os.environ['TILELANG_DISABLE_CACHE'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    import torch_npu  # noqa: F401
    import tilelang
    from tilelang.ascend.callback import register_ascend_postproc
    from tilelang.profiler.msprof import get_msprof_cache_flush_kernel
    from example_gemm import gemm

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    tilelang.disable_cache()
    blocks, size = args.blocks[0], args.size
    elements, tasks = size * size, (size // 256) ** 2
    args.output.mkdir(parents=True, exist_ok=True)
    props = torch.npu.get_device_properties(args.device)
    if not str(props.name).startswith('Ascend950'):
        raise RuntimeError("This diagnostic's default clock conversion is for Ascend950")
    a, b = [(torch.randn(size, size, device='npu') * 0.125).to(torch.bfloat16) for _ in range(2)]
    reference = (a.cpu().float() @ b.cpu().float().T).to(torch.bfloat16)
    # Four BF16 elements occupy one int64 word. Output and records do not overlap.
    backing = torch.empty(elements + blocks * RECORD_WORDS * 4, dtype=torch.bfloat16, device='npu')
    out = backing[:elements].view(size, size)
    records = backing[elements:].view(torch.int64).view(blocks, RECORD_WORDS)
    out.fill_(float('nan'))
    records.fill_(-1)
    # Compile/execute the flush helper before registering the Cube-only callback.
    flush_buffer = torch.empty((1 << 30) // 4, dtype=torch.int32, device='npu')
    flush = get_msprof_cache_flush_kernel(flush_buffer.numel(), 'int32')

    def postproc(code, target):
        if '_tilelang_profiler_cache_flush' in code:
            return code
        (args.output / 'original.asc').write_text(code)
        instrumented = instrument_source(code, elements)
        (args.output / 'instrumented.asc').write_text(instrumented)
        return instrumented

    register_ascend_postproc(postproc)
    kernel = tilelang.compile(gemm(size, size, size, dtype='bfloat16', out_dtype='bfloat16',
                                  MIXED=False, num_blocks=blocks),
                              target={'kind': 'ascend', 'arch': 'dav-3510'},
                              execution_backend='tvm_ffi', out_idx=[],
                              pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['-g']})
    if 'TILELANG_BLOCK_MAPPING_DIAGNOSTIC' not in kernel.get_kernel_source():
        raise RuntimeError("Uninstrumented cached kernel; refusing to launch")
    flush(flush_buffer)
    torch.npu.synchronize()
    kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
    raw = records.cpu().tolist()
    rows = decode_records(raw, blocks, tasks, args.ticks_per_us)
    write_csv(args.output / 'records.csv', rows)
    metadata = dict(device=str(props), size=size, blocks=blocks, tasks=tasks,
                    tilelang_path=tilelang.__file__, ticks_per_us=args.ticks_per_us,
                    raw_columns=list(COLUMNS), raw_records=raw, correctness='passed',
                    note='Instrumented single call, no warmup/profiler. Timestamps exclude asc_init and record publication; not baseline timing.')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'\n===== blocks={blocks}, tiles={tasks}, Cube={props.cube_core_num}, correctness=passed =====', flush=True)
    print('System counter: 1000 ticks/us on Ascend950 by default; raw core IDs preserved.', flush=True)
    for core in sorted({row['physical_core_raw'] for row in rows}):
        group = sorted((row for row in rows if row['physical_core_raw'] == core), key=lambda row: row['start_cycles'])
        entries = [f"block={r['logical_block']} tiles={r['issued_output_tiles']} "
                   f"start={r['start_us']:.3f} end={r['end_us']:.3f} duration={r['duration_us']:.3f}us"
                   + (f" EXIT_CORE={r['physical_core_exit_raw']}" if r['core_changed'] else '') for r in group]
        print(f'physical_core_raw={core}: ' + ' | '.join(entries), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', nargs='+', type=int, default=[16, 28, 29, 32, 56])
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ticks-per-us', type=float, default=1000.0)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--output', type=Path, default=Path('block-mapping'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(b <= 0 for b in args.blocks) or args.size < 256 or args.size % 256:
        parser.error('Blocks must be positive and size must be a positive multiple of 256')
    if not all(math.isfinite(value) and value > 0 for value in (args.ticks_per_us, args.timeout)):
        parser.error('Clock conversion and timeout must be positive and finite')
    args.output = args.output.resolve()
    if args.worker:
        if len(args.blocks) != 1:
            parser.error('A worker handles exactly one blocks configuration')
        run_case(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for blocks in dict.fromkeys(args.blocks):
        # Refuse stale outputs; use another output folder when rerunning.
        destination = args.output / f'blocks-{blocks}'
        destination.mkdir(exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--blocks', str(blocks),
                   '--size', str(args.size), '--device', str(args.device), '--seed', str(args.seed),
                   '--ticks-per-us', str(args.ticks_per_us), '--output', str(destination)]
        try:
            with (destination / 'run.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            print((destination / 'run.log').read_text(), flush=True)
            if result.returncode:
                raise RuntimeError(f'blocks={blocks} failed; see {destination / "run.log"}')
            with (destination / 'records.csv').open() as handle:
                rows.extend(csv.DictReader(handle))
            write_csv(args.output / 'records.csv', rows)
        except subprocess.TimeoutExpired:
            print(f'blocks={blocks} timed out; see {destination / "run.log"}', flush=True)
            raise
    print(f'Combined records: {args.output / "records.csv"}', flush=True)


if __name__ == '__main__':
    main()
