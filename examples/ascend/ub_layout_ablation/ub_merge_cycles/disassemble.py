"""Extract the measured executable's device ELFs; never certify ISA equivalence."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command, log):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}: {command}; see {log}")
    return result.stdout


def disassemble(build, msobjdump="msobjdump", objdump="llvm-objdump"):
    directory = build / "disasm"
    directory.mkdir(exist_ok=True)
    # Unique extraction directory prevents stale ELFs from earlier builds.
    extracted = Path(tempfile.mkdtemp(prefix="elfs-", dir=directory))
    manifest = {"status": "failed", "review": "pending", "device_elfs": []}
    try:
        demo = build / "demo"
        manifest["demo_sha256"] = sha256(demo)
        measurement = build / "measurement.json"
        manifest["measurement_match"] = "unknown_legacy_binary"
        if measurement.exists():
            expected = json.loads(measurement.read_text())["demo_sha256"]
            if expected != manifest["demo_sha256"]:
                raise RuntimeError("demo hash differs from measured binary; rerun cycles for this binary")
            manifest["measurement_match"] = "matched"
        for name, tool, version_option in (
                ("msobjdump", msobjdump, "-h"), ("objdump", objdump, "--version")):
            resolved = shutil.which(tool)
            if resolved is None:
                raise RuntimeError(f"Tool not found: {tool}")
            manifest[name] = resolved
            capture([resolved, version_option], directory / f"{name}-version.txt")
        capture([msobjdump, "--dump-elf", str(demo), "--verbose"], directory / "meta.txt")
        capture([msobjdump, "--list-elf", str(demo)], directory / "elf-list.txt")
        capture([msobjdump, "--extract-elf", str(demo), "--out-dir", str(extracted)],
                directory / "extract.log")
        for path in sorted(extracted.rglob("*")):
            if not path.is_file():
                continue
            header = path.read_bytes()[:20]
            if len(header) < 20 or header[:4] != b"\x7fELF":
                continue
            byteorder = {1: "little", 2: "big"}.get(header[5])
            if byteorder is None:
                continue
            machine = int.from_bytes(header[18:20], byteorder)
            record = {"path": str(path), "sha256": sha256(path), "elf_machine": machine}
            manifest["device_elfs"].append(record)
            # Do not mistake common host CPU code for NPU disassembly.
            if machine in (3, 40, 62, 183):
                record["status"] = "host_elf_skipped"
                continue
            try:
                capture([objdump, "--file-headers", "--section-headers", "--syms", str(path)],
                        Path(str(path) + ".info.txt"))
                asm = capture([objdump, "-d", "--demangle", str(path)],
                              Path(str(path) + ".asm"))
                has_instructions = re.search(r"^\s*[0-9a-fA-F]+:\s+\S", asm, re.MULTILINE)
                if not has_instructions or "<unknown>" in asm.lower():
                    record["status"] = "unsupported_or_incomplete"
                else:
                    record["status"] = "output_generated_unreviewed"
                hints = [f"{i}: {line}" for i, line in enumerate(asm.splitlines(), 1)
                         if re.search(r"bank_conflict|\blds|\bsts|clock|timestamp", line, re.I)]
                Path(str(path) + ".hints.txt").write_text("\n".join(hints) + "\n")
            except RuntimeError as error:
                record.update(status="failed", error=str(error))
        devices = [r for r in manifest["device_elfs"] if r["status"] != "host_elf_skipped"]
        if devices and all(r["status"] == "output_generated_unreviewed" for r in devices):
            manifest["status"] = "output_generated_unreviewed"
        else:
            manifest["error"] = "No complete device disassembly; inspect per-ELF status/logs"
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        manifest["error"] = str(error)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    review = directory / "review.md"
    if not review.exists():
        review.write_text(
            "# Manual ISA review (pending)\n\n"
            "| item | observation | evidence: ELF hash, function and instruction address |\n"
            "| --- | --- | --- |\n"
            "| timed region and end-of-work ordering | unknown | |\n"
            "| dynamic UB load count | unknown | |\n"
            "| bytes per lane: b32, b64 or two b32 | unknown | |\n"
            "| lane mask and UB base alignment | unknown | |\n"
            "| address expression | unknown | |\n"
            "| arithmetic, dependencies and spill/reload | unknown | |\n\n"
            "A b64 instruction does not expose its internal request/entry boundaries.\n"
            "Review applies only to the executable hash in manifest.json.\n")
    return manifest["status"]
