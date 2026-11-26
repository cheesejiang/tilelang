"""Recompile saved real-kernel C++ and disassemble its device object."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid


def sources(root, variants):
    from run_bottleneck_study import variants as planned
    modes = {name: mode for name, mode, _, _ in planned()}
    result = []
    for name in variants:
        if name not in modes:
            raise ValueError(f"Unknown variant {name}; choose from {', '.join(modes)}")
        result.append((name, root / name / 'gemm_scale' / f'{modes[name]}.cpp'))
    return result


def compiler_options(arch, include_dirs, extra):
    # Same defaults as tilelang.contrib.bisheng.get_bisheng_compile_options.
    return ['-O2', '-fPIC', '-std=c++20', '-mllvm', '-cce-aicore-dcpreload-args=false',
            f'--npu-arch={arch}', *extra,
            *[flag for path in include_dirs for flag in ('-I', str(path))]]


def run(command, log):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}")
    return result.stdout


def decoded(asm):
    return (bool(re.search(r'^\s*[0-9a-fA-F]+:\s+\S', asm, re.M)) and
            not re.search(r'<(?:unknown|not available)>', asm, re.I))


def process(source, output, args):
    output.mkdir(parents=True)
    original = source.read_bytes()
    manifest = {'status': 'failed', 'source': str(source),
                'source_sha256': hashlib.sha256(original).hexdigest(),
                'provenance': 'Recompiled saved source, not a hash-verified measured binary.',
                'isa_review': 'pending', 'arch': args.arch}
    try:
        objdump = shutil.which(args.objdump)
        if not objdump:
            raise RuntimeError(f"Device disassembler not found: {args.objdump}")
        manifest['objdump'] = objdump
        if source.suffix in ('.cpp', '.asc'):
            compiler = args.compiler
            if compiler is None:
                home = os.environ.get('BISHENG_HOME')
                compiler = str(Path(home) / 'bin/bisheng') if home else 'bisheng'
            compiler = shutil.which(compiler)
            if not compiler:
                raise RuntimeError('bisheng not found; load CANN or pass --compiler')
            # Copy exact bytes, no volatile or other source transformations.
            asc = output / 'kernel.asc'
            asc.write_bytes(original)
            obj = output / 'kernel.o'
            includes = [Path(__file__).resolve().parents[3] / 'src', *args.include]
            cmd = [compiler, *compiler_options(args.arch, includes, args.compile_arg),
                   '--cce-aicore-only', str(asc), '-o', str(obj)]
            manifest['compile_command'] = cmd
            run(cmd, output / 'compile.log')
        else:
            obj = output / 'kernel.o'
            obj.write_bytes(original)
            manifest['provenance'] = 'Disassembling the supplied object; measurement identity unverified.'
        header = obj.read_bytes()[:20]
        if len(header) < 20 or header[:4] != b'\x7fELF' or header[5] not in (1, 2):
            raise RuntimeError('Input/compiler output is not an ELF object')
        machine = int.from_bytes(header[18:20], 'little' if header[5] == 1 else 'big')
        manifest.update(object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(), elf_machine=machine)
        if machine in (3, 40, 62, 183):
            raise RuntimeError('Host ELF supplied; extract the device ELF with msobjdump first')
        info_cmd = [objdump, '--file-headers', '--section-headers', '--syms', str(obj)]
        manifest['info_command'] = info_cmd
        run(info_cmd, output / 'device.info.txt')
        cmd = [objdump, '-d', '--demangle', str(obj)]
        manifest['disassemble_command'] = cmd
        asm = run(cmd, output / 'device.asm')
        manifest['status'] = 'generated_unreviewed' if decoded(asm) else 'unsupported_or_incomplete'
        hints = [f'{n}: {line}' for n, line in enumerate(asm.splitlines(), 1)
                 if re.search(r'main_kernel|\bLDS\b|\bSTS\b|\bIADD\b|\bFMUL\b', line, re.I)]
        (output / 'isa_hints.txt').write_text('\n'.join(hints) + '\n')
        (output / 'review.md').write_text(
            '# ISA review\n\nLocate main_kernel_simt_vf_0 in device.asm.\n\n'
            '| item | observation | instruction addresses |\n| --- | --- | --- |\n'
            '| actual LDS/STS widths | pending | |\n'
            '| dynamic loads/stores including loop trip counts | pending | |\n'
            '| same lane addresses, byte coverage and masks | pending | |\n'
            '| IADD, FMUL, spills and scheduling | pending | |\n\n'
            'Do not multiply warp instructions by lane count. Text matches are static, not dynamic counts.\n')
    except (OSError, RuntimeError) as error:
        manifest['error'] = str(error)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"{source}: {manifest['status']}; {output}", flush=True)
    return manifest['status'] == 'generated_unreviewed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/ub-bottleneck-study'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--variants', nargs='+', default=None,
                       help='e.g. p8 p8-scalar baseline baseline-scalar ownership')
    group.add_argument('--source', type=Path, help='one saved .cpp/.asc or extracted device ELF')
    parser.add_argument('--output', type=Path, help='default: ROOT/isa/<unique run>')
    parser.add_argument('--compiler', help='bisheng path, defaults to BISHENG_HOME or PATH')
    parser.add_argument('--objdump', default='llvm-objdump', help='dav-3510-capable device disassembler')
    parser.add_argument('--arch', default='dav-3510')
    parser.add_argument('-I', '--include', type=Path, action='append', default=[])
    parser.add_argument('--compile-arg', action='append', default=[],
                        help='extra original compile option, repeat as --compile-arg=-FLAG')
    parser.add_argument('--dry-run', action='store_true', help='resolve source paths without compiling')
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    try:
        inputs = ([(args.source.stem, args.source.expanduser().resolve())] if args.source else
                  sources(args.root, args.variants or ['p8', 'p8-scalar']))
        for _, source in inputs:
            if not source.is_file():
                raise ValueError(f'Source does not exist: {source}')
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        for name, source in inputs:
            print(f'{name}: {source}')
        return
    output = ((args.output.expanduser().resolve() if args.output else args.root / 'isa') /
              uuid.uuid4().hex)
    outcomes = [process(source, output / name, args) for name, source in inputs]
    if not all(outcomes):
        raise SystemExit('Some objects were not fully disassembled; inspect manifest.json and logs')


if __name__ == '__main__':
    main()
