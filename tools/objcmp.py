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
  pruned          some symbols exist on one side only, but every one of them is
                  an inline or template copy (weak, GNU-unique) or local to the
                  object, and nothing left on the other side refers to it. This
                  is what removing an unused #include does: the header's inline
                  functions and its namespace-scope statics stop being emitted
                  into a file that never called them. The functions both sides
                  have are compared as usual and must be no worse than
                  scheduling.
  inlining        a function's relocations differ, but only by library code --
                  std::, __gnu_cxx::, Eigen::, g2o::, cv::, Sophus::, memcpy and
                  friends, or section-relative constants -- being inlined on one
                  side and called on the other. Which of this project's own
                  functions run is unchanged. It is not bit-for-bit safe: GCC may
                  contract a*b+c into one FMA differently once a body is inlined,
                  so a numeric function in this class can differ in the last bit.
                  Listed by name so the numeric ones can be looked at.
  CHANGED         a function calls or references something different, a
                  function with external linkage appeared or vanished, or a
                  vanished symbol is still referenced. This is a real code change
                  and needs explaining.

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


def symbols(path):
    out = subprocess.run(["nm", "-C", "--defined-only", str(path)], capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+\s+(\S)\s+(.*)$", line)
        if m:
            table[m.group(2)] = m.group(1)
    return table


def referenced(path):
    out = subprocess.run(["objdump", "-r", "-C", str(path)], capture_output=True, text=True).stdout
    return {re.sub(r"\+0x[0-9a-f]+$", "", m.group(1).strip()) for m in re.finditer(r"R_\w+\s+(.*)", out)}


# Weak, weak-object and GNU-unique: comdat copies the linker deduplicates.
# Lower case: local to this object. Neither can be something another object
# was relying on this one to define.
def private_or_comdat(letter):
    return letter in "WVwvu" or letter.islower()


LIBRARY = ("std::", "__gnu_cxx::", "Eigen::", "g2o::", "cv::", "Sophus::", "boost::",
           "operator new", "operator delete")
RUNTIME = {"memcpy", "memmove", "memset", "memcmp", "strlen", "__cxa_atexit", "__dso_handle"}


def library(target):
    # A relocation target that belongs to library or runtime code, or is a
    # section-relative reference (.text, .rodata.cst16, ...) -- i.e. not one of
    # this project's own functions or objects.
    t = re.sub(r"^(void|bool|int|unsigned|float|double|char|auto)\b\s*", "", target)
    return (t.startswith(".") or t in RUNTIME or t.startswith(LIBRARY)
            or re.match(r"^[\w:<>,\s*&]+ (std|Eigen|__gnu_cxx|g2o|cv|Sophus|boost)::", t) is not None)


def classify(a, b):
    if a.read_bytes() == b.read_bytes():
        return "bit-identical", []
    ca, ra = disassembly(a)
    cb, rb = disassembly(b)
    why = []

    # The functions both sides have.
    common = set(ca) & set(cb)
    differing = [n for n in common if ca[n] != cb[n]]
    moved = [n for n in differing if ra.get(n) != rb.get(n)]
    # A static-initialisation function whose relocations only shrank (or only
    # grew) constructs fewer (more) namespace-scope objects and changes nothing
    # else -- which is exactly what dropping (adding) an #include whose header
    # defines statics does. Whether those objects were used anywhere is checked
    # below, with the one-sided symbols: any that is external, or still
    # referenced, is a real change.
    initialisers = [n for n in moved if n.startswith("_GLOBAL__sub_I_")
                    and ((ra.get(n, collections.Counter()) - rb.get(n, collections.Counter())) == collections.Counter()
                         or (rb.get(n, collections.Counter()) - ra.get(n, collections.Counter())) == collections.Counter())]
    moved = [n for n in moved if n not in initialisers]
    inlined, real = [], []
    for n in moved:
        delta = (ra.get(n, collections.Counter()) - rb.get(n, collections.Counter())) \
              + (rb.get(n, collections.Counter()) - ra.get(n, collections.Counter()))
        (inlined if all(library(t) for t in delta) else real).append(n)
    if real:
        return "CHANGED", ["relocations differ: " + n for n in real[:8]]
    differing = [n for n in differing if n not in initialisers and n not in inlined]
    cls = "scheduling" if differing else "reordered"
    if differing:
        why.append("%d functions scheduled differently, e.g. %s" % (len(differing), differing[0]))
    if inlined:
        cls = "inlining"
        why += ["library inlined on one side only: " + n for n in inlined[:6]]

    # Symbols only one side has.
    sa, sb = symbols(a), symbols(b)
    one_sided = [(n, sa.get(n) or sb.get(n), n in sa) for n in (set(sa) ^ set(sb))]
    if one_sided:
        refs_a, refs_b = referenced(a), referenced(b)
        bad = []
        for name, letter, in_a in one_sided:
            other_refs = refs_b if in_a else refs_a
            if not private_or_comdat(letter):
                bad.append("external %s only on one side: %s" % (letter, name))
            elif name in other_refs:
                bad.append("one-sided but still referenced: " + name)
        if bad:
            return "CHANGED", bad[:8]
        if cls == "reordered":
            cls = "pruned"
        why.append("%d inline/local symbols on one side only, none referenced" % len(one_sided))
    if initialisers:
        if not one_sided:
            return "CHANGED", ["static initialiser changed with no symbol removed: " + initialisers[0]]
        if cls == "reordered":
            cls = "pruned"
        why.append("static initialiser constructs %s objects: %s"
                   % ("fewer" if sum(rb.get(initialisers[0], {}).values()) < sum(ra.get(initialisers[0], {}).values())
                      else "more", initialisers[0]))
    return cls, why


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
    for cls in ("bit-identical", "reordered", "pruned", "scheduling", "inlining", "CHANGED"):
        print("%-14s %d" % (cls, counts.get(cls, 0)))
    return 1 if counts.get("CHANGED") else 0


if __name__ == "__main__":
    sys.exit(main())
