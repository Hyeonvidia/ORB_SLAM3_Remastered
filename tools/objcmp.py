#!/usr/bin/env python3
"""
Compares two build trees object file by object file, and says for each one how
far apart they are -- which is the question a refactor that claims to change
nothing has to answer.

  ./docker/run.sh -- python3 /workspace/tools/objcmp.py BEFORE_DIR AFTER_DIR

Four classes, from strongest to weakest evidence that nothing changed:

  bit-identical   the bytes are the same.
  reordered       every function compiles to the same instructions, but the
                  compiler emitted inline/template (comdat) sections in a
                  different order. Typical when an #include moves: the order in
                  which headers are seen changes the order in which their inline
                  functions are written out. The linker deduplicates comdat
                  groups, so their position in the .o is not observable.
  scheduling      some functions' instructions differ, but each one still
                  carries exactly the same relocations -- it calls and references
                  the same symbols. What changed is register allocation,
                  instruction scheduling or block placement, which is what an
                  optimiser does when a different set of inline definitions is
                  visible. Not observable either, but this is the class to look
                  at if a behavioural difference ever does turn up.
  CHANGED         a function calls or references something different. This is a
                  real code change and needs explaining.

Byte comparison alone is too strict for this project: moving one #include out of
common/Settings.hpp changed 13 of 42 object files while changing no function's
behaviour at all. Instruction comparison alone is too weak the other way -- an
unlinked .o shows "bl 0 <first symbol in section>" for every external call, so
the printed call targets are placeholders; only the relocation table says what
is really called.

Run inside the dev container (it needs binutils' objdump for the target).
"""
import collections
import hashlib
import re
import subprocess
import sys
from pathlib import Path


def disassembly(path):
    out = subprocess.run(["objdump", "-dr", "--no-show-raw-insn", "-C", str(path)],
                         capture_output=True, text=True).stdout
    code, rel = collections.defaultdict(list), collections.defaultdict(collections.Counter)
    cur = None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            cur = m.group(1)
            continue
        if cur is None or not line.strip() or line.startswith("Disassembly of section"):
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+R_\w+\s+(.*)$", line)
        if m:
            rel[cur][re.sub(r"\+0x[0-9a-f]+$", "", m.group(1).strip())] += 1
            continue
        ins = re.sub(r"^\s*[0-9a-f]+:\s*", "", line)
        ins = re.sub(r"<[^>]*>", "", ins)                 # placeholder symbol names
        ins = re.sub(r"\b0x[0-9a-f]+\b|\b[0-9a-f]{3,}\b", "N", ins)   # offsets
        code[cur].append(ins)
    digest = {k: hashlib.sha1("\n".join(v).encode()).hexdigest() for k, v in code.items()}
    return digest, rel


def classify(a, b):
    if a.read_bytes() == b.read_bytes():
        return "bit-identical", []
    ca, ra = disassembly(a)
    cb, rb = disassembly(b)
    if set(ca) != set(cb):
        diff = sorted(set(ca) ^ set(cb))
        return "CHANGED", ["only on one side: " + n for n in diff[:8]]
    differing = [n for n in ca if ca[n] != cb[n]]
    if not differing:
        return "reordered", []
    moved = [n for n in differing if ra.get(n) != rb.get(n)]
    if moved:
        return "CHANGED", ["relocations differ: " + n for n in moved[:8]]
    return "scheduling", ["%d functions, e.g. %s" % (len(differing), differing[0])]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for a in sorted(after.rglob("*.o")):
        rel = a.relative_to(after)
        b = before / rel
        if not b.is_file():
            rows.append((str(rel), "CHANGED", ["missing from " + str(before)]))
            continue
        cls, why = classify(b, a)
        rows.append((str(rel), cls, why))
    counts = collections.Counter(r[1] for r in rows)
    for name, cls, why in rows:
        if cls != "bit-identical":
            print("%-11s %s" % (cls, name))
            for w in why:
                print("            " + w[:150])
    print()
    for cls in ("bit-identical", "reordered", "scheduling", "CHANGED"):
        print("%-14s %d" % (cls, counts.get(cls, 0)))
    return 1 if counts.get("CHANGED") else 0


if __name__ == "__main__":
    sys.exit(main())
