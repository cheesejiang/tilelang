"""Single-call target for external msopprof; no nested Python profiler.

Run from the built TileLang checkout on the 9579. This is mechanism diagnosis,
not a replacement for the established multi-sample performance benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=['gemm_bf16', 'gemm_fp32', 'gemm_mixed_fp32', 'vecadd_simt', 'vecadd_simd'], default='gemm_bf16')
parser.add_argument('--blocks', type=int, required=True)
parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
parser.add_argument('--repo', type=Path, default=Path.cwd())
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
if args.blocks <= 0:
    parser.error('--blocks must be positive')
repo = args.repo.resolve()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
os.environ['TILELANG_DEFAULT_TARGET'] = 'ascend'
os.environ['TILELANG_EXECUTION_BACKEND'] = 'tvm_ffi'
os.environ['ASCEND_NPU_ARCH'] = 'dav-3510'
sys.path[:0] = [str(repo), str(repo/'examples/ascend')]
import torch
import torch_npu
import tilelang
from tilelang.contrib import bisheng
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.profiler.msprof import get_msprof_cache_flush_kernel

torch.npu.set_device(args.device)
torch.manual_seed(0)
full = args.profile == 'full'
if args.case.startswith('vecadd'):
    from example_vecadd import vector_add
    n = 2**30 if full else 2**20
    a, b = [(torch.randn(n, device='npu') * 0.125) for _ in range(2)]
    func = vector_add(n, args.case.removeprefix('vecadd_'), num_blocks=args.blocks)
    shape = [n]
else:
    from example_gemm import gemm
    n = 8192 if full else 1024
    dtype = torch.bfloat16 if args.case == 'gemm_bf16' else torch.float32
    a, b = [(torch.randn(n, n, device='npu') * 0.125).to(dtype) for _ in range(2)]
    func = gemm(n,n,n,dtype='bfloat16' if dtype == torch.bfloat16 else 'float32',
                out_dtype='bfloat16' if dtype == torch.bfloat16 else 'float32',
                MIXED=args.case == 'gemm_mixed_fp32',num_blocks=args.blocks)
    shape = [n,n,n]
matmul = getattr(torch.npu, 'matmul', None)
if matmul is not None and hasattr(matmul, 'allow_hf32'):
    matmul.allow_hf32 = False
kernel = tilelang.compile(func, target={'kind':'ascend','arch':'dav-3510'},
    execution_backend='tvm_ffi', out_idx=-1,
    pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS:['-g']})
source = kernel.get_kernel_source()
(args.output/'kernel.asc').write_text(source)
compiler = bisheng.find_bisheng_path()
options = bisheng.get_bisheng_compile_options('dav-3510',['-I'+str(TILELANG_TEMPLATE_PATH),'-g'])
commands = [
    [compiler,*options,'--cce-aicore-only',str(args.output/'kernel.asc'),'-o',str(args.output/'kernel.rel.o')],
    [bisheng.find_ld_lld_path(),*bisheng.get_aibin_linker_options(),str(args.output/'kernel.rel.o'),'-o',str(args.output/'kernel.aibin')],
]
for index, command in enumerate(commands):
    result = subprocess.run(command,capture_output=True,text=True)
    (args.output/f'build-{index}.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
objdump = Path(compiler).resolve().with_name('llvm-objdump')
if not objdump.exists():
    objdump = shutil.which('llvm-objdump')
if objdump:
    result = subprocess.run([str(objdump),'-d','-l',str(args.output/'kernel.aibin')],capture_output=True,text=True)
    (args.output/'disassembly.txt').write_text(result.stdout+result.stderr)
    result.check_returncode()
else:
    raise RuntimeError('llvm-objdump not found')
metadata = dict(case=args.case,shape=shape,blocks=args.blocks,device=str(torch.npu.get_device_properties(args.device)),
                tilelang=tilelang.__file__,ascend_home=os.environ.get('ASCEND_HOME_PATH'),
                calls=1,debug_compile=True,commands=commands,
                note='One cold-cache diagnostic call, no warmup or nested profiler; not a performance baseline')
(args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
flush_buffer = torch.empty((1<<30)//4, dtype=torch.int32, device='npu')
flush = get_msprof_cache_flush_kernel(flush_buffer.numel(),'int32')
flush(flush_buffer)
torch.npu.synchronize()
# The external profiler filters main_kernel, excluding initialization/flush.
out = kernel(a,b)
torch.npu.synchronize()
assert torch.isfinite(out).all().item()
print(f'Completed {args.case}, blocks={args.blocks}; artifacts: {args.output}', flush=True)
