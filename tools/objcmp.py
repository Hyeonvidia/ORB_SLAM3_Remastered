#!/usr/bin/env python3
"""
Compares two build trees object file by object file, and says for each one how
far apart they are -- which is the question a refactor that claims to change
nothing has to answer.

  ./docker/run.sh -- python3 /workspace/tools/objcmp.py BEFORE_DIR AFTER_DIR
  ./docker/run.sh -- python3 /workspace/tools/objcmp.py \\
      --rename 'ORB_SLAM3::Old::Name=ORB_SLAM3::New::Name' BEFORE_DIR AFTER_DIR

Exit status 1 if anything is CHANGED, 2 if either side has no object files.
tools/objcmp_tests/run.sh checks the tool against cases it must and must not
flag.

Seven classes, from strongest to weakest evidence that nothing changed:

  bit-identical   the bytes are the same.
  reordered       every function has the same instructions, with the same
                  targets at the same places, and every piece of data has the
                  same contents; the compiler only emitted inline/template
                  (comdat) sections in a different order. Typical when an
                  #include moves: the order in which headers are seen changes
                  the order in which their inline functions are written out. The
                  linker deduplicates comdat groups, so their position in the .o
                  is not observable.
  pruned          as reordered, but some symbols exist on one side only, and
                  every one of them is an inline or template copy (weak,
                  GNU-unique) or local to the object that nothing left on the
                  other side refers to; or the static initialiser constructs
                  fewer objects. This is what removing an unused #include does:
                  the header's inline functions and namespace-scope statics stop
                  being emitted into a file that never used them.
  moved           a definition with external linkage left this object for
                  another one, or arrived from one, and is the same where it now
                  lives -- the same instructions and targets, and the same local
                  functions and data behind them. An object file that exists on
                  one side only is in this class when everything it defines is
                  accounted for that way; one that only changed directory or
                  name is paired by file name or by its bytes.
  scheduling      some functions' instructions differ, but each still calls and
                  references the same symbols the same number of times (the
                  order may differ) and uses the same immediate values. What
                  changed is register allocation, instruction scheduling or
                  block placement, which is what an optimiser does when a
                  different set of inline definitions is visible. Not observable
                  either, but this is the class to look at if a behavioural
                  difference ever does turn up.
  inlining        a function's instructions and targets differ, but its targets
                  only by library code -- std::, Eigen::, g2o::, cv::, the C and
                  C++ runtime -- being inlined on one side and called on the
                  other, together with the constants and strings that library
                  code brings along. Which of this project's own functions run is
                  unchanged. It is not bit-for-bit safe: its immediates are not
                  compared, and GCC may contract a*b+c into one FMA differently
                  once a body is inlined, so a numeric function in this class can
                  differ in the last bit. Listed by name so they can be looked at.
  CHANGED         a function calls or references something different, uses a
                  different immediate, or data differs; a definition with
                  external linkage appeared or vanished without turning up
                  unchanged in another object; a vanished symbol is still
                  referenced; the static initialiser does more at startup; or
                  a catch clause matches a different type. A real code change,
                  which needs explaining.

What is compared, for every function both sides have:

  - its instruction stream, with alignment nops dropped and each branch inside
    the function written as the index of the instruction it lands on;
  - the targets it calls or references, in order. Those are the relocations --
    an unlinked .o shows "bl 0 <first symbol in section>" for every external
    call, so only the relocation table says what is really called -- plus the
    calls the assembler resolved itself, to functions local to the object. A
    reference to something local is resolved by content: a string literal to
    the string, a pooled constant to its bytes, a static to its symbol and its
    bytes;
  - the values it uses as immediate operands, as a set -- what a changed
    threshold or constant changes. Addressing is left out: memory offsets,
    64-bit pointer arithmetic and anything relative to the stack or frame
    pointer all move with register allocation, and so does how often one
    constant is materialised.

And for the object: the contents of every data symbol both sides have, the
static initialiser, and the exception types each function's catch clauses
match.

GCC numbers the specialised copies it makes of a function ("[clone
.constprop.0]") in the order it meets them, so those are renumbered by content
first.

Known limits, each with a case in tools/objcmp_tests. An anonymous table -- a
switch's jump table -- is compared through the instructions that index it, not
its bytes, so two cases swapping bodies can pass as scheduling; so can a
function that uses a value it already used elsewhere in place of another. With
-fsection-anchors, several statics are reached through one relocation and an
offset in the instruction; the offset is compared as an immediate, and the
relocation resolves to whichever static sits at the anchor, so reordering two
statics shows as CHANGED even though nothing changed. Written for aarch64
objdump output.

--rename OLD=NEW (repeatable) rewrites OLD to NEW in every demangled name read
from BEFORE, so a commit that renames a symbol can be checked for doing nothing
else. It is a plain substring replacement: give whole names, or it renames more
than meant.

Run inside the dev container (it needs binutils' objdump for the target).
"""
import collections
import hashlib
import re
import subprocess
import sys
from pathlib import Path

CLASSES = ("bit-identical", "reordered", "pruned", "moved", "scheduling", "inlining", "CHANGED")
INITIALISER = "_GLOBAL__sub_I_"

LIBRARY = ("std::", "__gnu_cxx::", "__cxxabiv1::", "Eigen::", "g2o::", "cv::", "Sophus::", "boost::",
           "DBoW2::", "DUtils::", "pangolin::", "picojson::", "operator new", "operator delete",
           "typeinfo for std::", "vtable for std::", "VTT for std::")


def sha(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


class Fn:
    """One function: its instruction stream and what it points at."""

    def __init__(self, shape, targets, immediates):
        self.shape = shape                                  # hash of the instructions
        self.targets = tuple(targets)                       # what it calls and references, in order
        self.calls = collections.Counter(targets)           # the same, without the order
        # The values it uses as immediates, as a set: register allocation may
        # materialise one constant once or before every use.
        self.immediates = frozenset(immediates)


class Obj:
    """One object file, read lazily. Names read from it go through the renames."""

    def __init__(self, path, renames=()):
        self.path = path
        self.renames = renames
        self._fns = self._syms = self._refs = self._table = self._bytes = self._relocs = None
        self._digests, self._deep, self._visiting = {}, {}, set()

    def run(self, *cmd, rename=True):
        out = subprocess.run(list(cmd) + [str(self.path)], capture_output=True, text=True).stdout
        if rename:
            for old, new in self.renames:
                out = out.replace(old, new)
        # One static initialiser per object, named after whichever global the
        # compiler saw first; the name says nothing about what it does.
        out = re.sub(r"global constructors keyed to .*?(?=>:$|$)", INITIALISER, out, flags=re.M)
        return re.sub(INITIALISER + r"[^\s>)]*", INITIALISER, out)

    # -- functions -----------------------------------------------------------

    def functions(self):
        if self._fns is not None:
            return self._fns
        bodies, patched, cur = collections.defaultdict(list), collections.defaultdict(dict), None
        for line in self.run("objdump", "-dr", "--no-show-raw-insn", "-C").splitlines():
            m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
            if m:
                cur = m.group(1)
                continue
            if cur is None or not line.strip() or line.startswith("Disassembly of section"):
                continue
            m = re.match(r"^\s+([0-9a-f]+):\s+R_\w+\s+(.*)$", line)
            if m:
                patched[cur][int(m.group(1), 16)] = self.target(m.group(2).strip())
                continue
            m = re.match(r"^\s*([0-9a-f]+):\s*(.*)$", line)
            if m:
                bodies[cur].append((int(m.group(1), 16), m.group(2)))
        fns = {name: self.function(name, body, patched[name]) for name, body in bodies.items()}
        self._fns = canonical_clones(fns)
        return self._fns

    def function(self, name, body, patched):
        # Alignment nops come and go with where the function starts; they do
        # nothing, so they are dropped and branches are written as the index of
        # the instruction they land on, which does not move with them.
        kept = [(addr, text) for addr, text in body
                if addr in patched or not re.match(r"^nop\b", text)]
        index, j = {}, 0
        for addr, _ in body:
            while j < len(kept) and kept[j][0] < addr:
                j += 1
            index[addr] = j
        lo, hi = (body[0][0], body[-1][0]) if body else (0, -1)
        lines, targets, imms = [], [], collections.Counter()
        for addr, text in kept:
            text = re.sub(r"\s+//.*$", "", text)          # decoded duplicate of an operand
            if addr in patched:
                targets.append(patched[addr])
                text = re.sub(r"\b[0-9a-f]+ <.*>$", "<@>", text)
                text = re.sub(r"#0x0\b|#0\b", "#@", text)   # the field the relocation fills
            else:
                m = re.search(r"\b([0-9a-f]+) <(.*)>$", text)
                if m:
                    dest = int(m.group(1), 16)
                    if lo <= dest <= hi:
                        text = text[:m.start()] + "<.%d>" % index.get(dest, -1)
                    else:                                   # a call the assembler resolved
                        targets.append("local " + m.group(2))
                        text = text[:m.start()] + "<@>"
                imms.update(values(text))
            lines.append(text)
        return Fn(sha("\n".join(lines)), targets, imms)

    # -- symbols and sections ------------------------------------------------


    def symbols(self):
        # name -> (nm letter, value, size)
        if self._syms is None:
            self._syms = {}
            for line in self.run("nm", "-C", "-S", "--defined-only").splitlines():
                m = re.match(r"^([0-9a-f]+)\s+(?:([0-9a-f]+)\s+)?(\S)\s+(.*)$", line)
                if m and not m.group(4).startswith("$"):
                    self._syms[m.group(4)] = (m.group(3), int(m.group(1), 16), int(m.group(2) or "0", 16))
        return self._syms

    def referenced(self):
        if self._refs is None:
            self._refs = {re.sub(r"[+-]0x[0-9a-f]+$", "", m.group(1).strip())
                          for m in re.finditer(r"R_\w+\s+(.*)", self.run("objdump", "-r", "-C"))}
        return self._refs

    def table(self):
        # Every symbol that sits in a section: [(section, value, size, name)]
        if self._table is None:
            self._table, self._code_names = [], set()
            for line in self.run("objdump", "-t", "-C").splitlines():
                m = re.match(r"^([0-9a-f]+)\s(.{7})\s(\S+)\s+([0-9a-f]+)\s+(?:\.hidden\s+)?(.*)$", line)
                if m and not re.search(r"[df]", m.group(2)[5:]) and not m.group(5).startswith("$"):
                    self._table.append((m.group(3), int(m.group(1), 16), int(m.group(4), 16), m.group(5)))
                    if m.group(2)[6] == "F":
                        self._code_names.add(m.group(5))
        return self._table

    def code_names(self):
        # Every symbol the symbol table calls a function.
        self.table()
        return self._code_names

    def section_of(self, name):
        for section, _, _, sym in self.table():
            if sym == name:
                return section
        return None

    def section_bytes(self, section):
        # The initial contents of one section, or None if there is no such
        # section or more than one of that name.
        if self._bytes is None:
            self._bytes, cur = {}, None
            for line in self.run("objdump", "-s", rename=False).splitlines():
                m = re.match(r"^Contents of section (.*):$", line)
                if m:
                    cur = m.group(1)
                    self._bytes[cur] = None if cur in self._bytes else bytearray()
                    continue
                m = re.match(r"^ ([0-9a-f]+) ([0-9a-f ]{35})", line)
                if m and cur is not None and self._bytes[cur] is not None:
                    off = int(m.group(1), 16)
                    chunk = bytes.fromhex(m.group(2).replace(" ", ""))
                    self._bytes[cur][off:off + len(chunk)] = chunk
        data = self._bytes.get(section)
        return None if data is None else bytes(data)

    def relocations(self):
        # section -> [(offset, relocation type, raw target)]
        if self._relocs is None:
            self._relocs, cur = collections.defaultdict(list), None
            for line in self.run("objdump", "-r", "-C").splitlines():
                m = re.match(r"^RELOCATION RECORDS FOR \[(.*)\]:$", line)
                if m:
                    cur = m.group(1)
                    continue
                m = re.match(r"^([0-9a-f]+)\s+(R_\w+)\s+(.*)$", line)
                if m and cur is not None:
                    self._relocs[cur].append((int(m.group(1), 16), m.group(2), m.group(3).strip()))
        return self._relocs

    # -- what a reference points at ------------------------------------------

    def target(self, raw):
        # A named symbol stays as printed, with its offset into that symbol. A
        # section-relative reference -- how the assembler writes one to
        # anything local to the object -- is resolved by content: a string
        # literal to the string, a pooled constant to its bytes, anything else
        # to the local symbol at that address, with that symbol's contents for
        # data. Only an address with no symbol is left as the bare section.
        m = re.match(r"^(.*?)(?:([+-])0x([0-9a-f]+))?$", raw)
        name = m.group(1)
        addend = int(m.group(3) or "0", 16) * (-1 if m.group(2) == "-" else 1)
        if not name.startswith("."):
            return raw
        data = self.section_bytes(name)
        if data is not None and re.search(r"\.str1\.\d+$", name):
            end = data.find(b"\0", addend)
            return "string %r" % data[addend:end if end >= 0 else len(data)]
        cst = re.search(r"\.cst(\d+)$", name)
        if data is not None and cst:
            return "constant " + data[addend:addend + int(cst.group(1))].hex()
        for section, value, size, sym in self.table():
            if section == name and value <= addend < value + max(size, 1):
                off = "+0x%x" % (addend - value) if addend != value else ""
                if section.startswith((".text", ".init", ".fini")):
                    return "local " + sym + off
                return "%s%s %s" % (sym, off, self.digest(section, value, size))
        return name

    def digest(self, section, value, size):
        # The contents of [value, value + size) in a section: its bytes and the
        # relocations inside it, each resolved like any other.
        key = (section, value, size)
        if key in self._digests:
            return self._digests[key]
        if key in self._visiting:                         # refers back to itself
            return "cycle"
        self._visiting.add(key)
        data = self.section_bytes(section)
        relocs = sorted((off - value, kind, self.target(t)) for off, kind, t in self.relocations().get(section, [])
                        if value <= off < value + size)
        self._visiting.discard(key)
        self._digests[key] = sha(repr((None if data is None else data[value:value + size], relocs)))
        return self._digests[key]

    def content(self, name):
        # What a definition is, independent of which object it sits in: for a
        # function, its instructions and its targets with every local function
        # it reaches folded in, so that a static helper that moved along with
        # it is compared too; for data, its bytes and relocations.
        if name in self._deep:
            return self._deep[name]
        fns = self.functions()
        if name in fns:
            if name in self._visiting:
                return "cycle"
            self._visiting.add(name)
            fn = fns[name]
            deep = []
            for t in fn.targets:
                callee = re.sub(r"\+0x[0-9a-f]+$", "", t[len("local "):]) if t.startswith("local ") else None
                deep.append(self.content(callee) if callee in fns else t)
            self._visiting.discard(name)
            caught = sorted(self.catches().get(unnumbered(name), {}).items())
            result = sha(repr(("code", fn.shape, deep, sorted(fn.immediates), caught)))
        else:
            letter, value, size = self.symbols().get(name, ("?", 0, 0))
            section = self.section_of(name)
            if section is None or letter in "Bb":         # no bytes to compare
                result = sha(repr(("bss", size)))
            else:
                result = sha(repr(("data", size, self.digest(section, value, size))))
        self._deep[name] = result
        return result

    def reached(self, names):
        # Every local function reachable from NAMES through calls and references.
        fns, seen, todo = self.functions(), set(), list(names)
        while todo:
            n = todo.pop()
            if n in seen or n not in fns:
                continue
            seen.add(n)
            todo += [re.sub(r"\+0x[0-9a-f]+$", "", t[len("local "):]) for t in fns[n].targets
                     if t.startswith("local ")]
        return seen

    def catches(self):
        # The types each function's catch clauses match. Each .eh_frame entry
        # points at its function and at where the function's entry in
        # .gcc_except_table starts; the type references from there to the next
        # entry's start are that function's.
        if getattr(self, "_catches", None) is not None:
            return self._catches
        rel = self.relocations()
        owner, last = {}, None
        for _, _, t in sorted(rel.get(".eh_frame", [])):
            m = re.match(r"^(.*?)(?:\+0x([0-9a-f]+))?$", t)
            if m.group(1).startswith(".gcc_except_table"):
                if last is not None:
                    owner[(m.group(1), int(m.group(2) or "0", 16))] = last
            else:
                last = self.target(t)
        out = collections.defaultdict(collections.Counter)
        for section, relocs in rel.items():
            if section.startswith(".gcc_except_table"):
                starts = sorted(a for s2, a in owner if s2 == section)
                for off, _, t in relocs:
                    start = max((a for a in starts if a <= off), default=None)
                    fn = owner.get((section, start), section)
                    fn = unnumbered(fn[len("local "):] if fn.startswith("local ") else fn)
                    out[fn][re.sub(r"^DW\.ref\.", "", t)] += 1
        self._catches = dict(out)
        return self._catches


CLONE = re.compile(r"\[clone \.\w+\.(\d+)\]")


def unnumbered(name):
    return CLONE.sub(lambda m: m.group(0).replace(m.group(1), "#"), name)


def canonical_clones(fns):
    # GCC numbers the specialised copies it makes of a function ("[clone
    # .constprop.0]", ".isra.1") in the order it makes them, which follows the
    # order of the definitions in the file. Moving one definition past another
    # swaps the numbers without changing either copy. Renumber each set of
    # copies by what they contain, so that the numbers mean the same thing on
    # both sides.
    def local(t):
        return "local " + unnumbered(t[len("local "):]) if t.startswith("local ") else t

    groups = collections.defaultdict(list)
    for name in fns:
        if CLONE.search(name):
            groups[unnumbered(name)].append(name)
    rename = {}
    for key, names in groups.items():
        names.sort(key=lambda n: (fns[n].shape, sorted(fns[n].immediates),
                                  [local(t) for t in fns[n].targets]))
        for i, n in enumerate(names):
            rename[n] = "%s #%d" % (key, i)
    if not rename:
        return fns

    def retarget(t):
        # A local copy is named either way: "local NAME" when the assembler
        # resolved the call or the relocation is against the section, plain
        # NAME when the relocation is against the symbol.
        prefix = "local " if t.startswith("local ") else ""
        m = re.match(r"^(.*?)((?:\+0x[0-9a-f]+)?)$", t[len(prefix):])
        return prefix + rename.get(m.group(1), m.group(1)) + m.group(2)

    return {rename.get(n, n): Fn(f.shape, [retarget(t) for t in f.targets], f.immediates)
            for n, f in fns.items()}


def values(text):
    # The immediates of one instruction that are values the code computes
    # with -- what a changed threshold or constant would change -- as opposed
    # to addressing: memory offsets, 64-bit pointer arithmetic, and anything
    # relative to the stack or frame pointer, which register allocation moves
    # (holding this+48 in a register shifts every offset after it by 48).
    op = text.split(None, 1)
    if len(op) < 2 or re.search(r"\b(sp|x29|w29)\b", op[1]) or re.match(r"^(ld|st|prfm)", op[0]):
        return []
    if op[0] in ("add", "sub", "adds", "subs") and op[1].startswith("x"):
        return []
    out = []
    for imm in re.findall(r"#(-?[0-9a-fx.e+]+)", re.sub(r"\[.*?\]|<.*>", "", op[1])):
        try:
            out.append(str(int(imm, 0)))                    # 0xc8 and 200 are the same
        except ValueError:
            out.append(imm)
    return out


def strong(letter):
    # A definition another object may be relying on: global and not a comdat
    # copy the linker deduplicates.
    return letter.isupper() and letter not in "WVU"


class Tree:
    """Every object file of one build, and which of them defines each name."""

    def __init__(self, root, renames=()):
        self.objs = {p.relative_to(root): Obj(p, renames) for p in sorted(root.rglob("*.o"))}
        self.defines = collections.defaultdict(list)
        for rel, obj in self.objs.items():
            for name, (letter, _, _) in obj.symbols().items():
                if strong(letter):
                    self.defines[name].append(rel)

    def find(self, name, content, exclude):
        # Another object defining NAME with the same content, or None.
        for rel in self.defines.get(name, []):
            if rel != exclude and self.objs[rel].content(name) == content:
                return rel
        return None


class Kinds:
    """Tells this project's code from library code, for the inlining class."""

    def __init__(self, before, after):
        self.project = set(before.defines) | set(after.defines)

    def kind(self, target):
        if target.startswith(("constant ", "string ")):
            return "literal"
        t = target[len("local "):] if target.startswith("local ") else target
        t = re.sub(r"\s[0-9a-f]{16}$", "", t)               # a data digest
        t = re.sub(r"[+-]0x[0-9a-f]+$", "", t)
        bare = qualified_name(t)
        if t.startswith("."):
            return "library"                                # an anonymous table
        if bare.startswith("ORB_SLAM3::") or t in self.project:
            return "project"
        if bare.startswith(LIBRARY) or t.startswith(LIBRARY):
            # A library template instantiated over this project's types -- a
            # comparator, a functor -- runs this project's code.
            return "library-over-project" if "ORB_SLAM3::" in t else "library"
        if "(" not in t and "::" not in t:
            return "library"                                # C and C++ runtime
        return "project"


def qualified_name(t):
    # The name a demangled symbol is for, without its return type or
    # parameters: the last word before the parameter list, counting only what
    # is outside template brackets.
    depth, word, words = 0, "", []
    for c in t:
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif depth == 0 and c == "(":
            break
        elif depth == 0 and c == " ":
            words.append(word)
            word = ""
            continue
        word += c
    words.append(word)
    return words[-1] if words[-1] else t


def worst(*classes):
    return max(classes, key=CLASSES.index)


def compare_functions(a, b, kinds):
    # -> (real changes, differing, inlined)
    fa, fb = a.functions(), b.functions()
    real, differing, inlined = [], [], []
    for n in sorted((set(fa) & set(fb)) - {INITIALISER}):
        x, y = fa[n], fb[n]
        if x.shape == y.shape:
            if x.targets != y.targets:
                real.append("same instructions, different targets: " + n)
            continue
        if x.calls == y.calls:
            if x.immediates != y.immediates:
                real.append("different immediates: " + n)
            else:
                differing.append(n)
            continue
        gone, came = x.calls - y.calls, y.calls - x.calls
        k = {t: kinds.kind(t) for t in list(gone) + list(came)}
        calls = [t for t in k if k[t] != "literal"]
        over = [t for t in calls if k[t] == "library-over-project"]
        if (calls and all(k[t] in ("library", "library-over-project") for t in calls)
                and not (any(t in gone for t in over) and any(t in came for t in over))):
            # An inlined library body brings its own pooled constants and
            # strings, so literals may come and go -- but only alongside library
            # code doing the same. A library template over this project's types
            # may be inlined or called, but not swapped for another.
            inlined.append(n)
        else:
            real.append("different targets: " + n)
    return real, differing, inlined


def classify(a, b, before, after, rel_a, rel_b, kinds):
    if a.path.read_bytes() == b.path.read_bytes():
        return "bit-identical", []
    real, differing, inlined = compare_functions(a, b, kinds)
    why = []

    # Data both sides define -- constants, tables, vtables, initial values.
    sa, sb = a.symbols(), b.symbols()
    fa, fb = a.functions(), b.functions()
    for n in sorted((set(sa) & set(sb)) - set(fa) - set(fb) - a.code_names() - b.code_names()):
        if not n.startswith("."):
            if a.content(n) != b.content(n):
                real.append("data differs: " + n)

    # What each function's catch clauses match.
    ca, cb = a.catches(), b.catches()
    for n in sorted(set(ca) | set(cb)):
        if ca.get(n) != cb.get(n):
            real.append("catches different exception types: " + n)

    # The static initialiser: running less at startup is what dropping an
    # unused #include does; running more, or something else, is a change.
    ia, ib = fa.get(INITIALISER), fb.get(INITIALISER)
    fewer = False
    if ib and not ia:
        real.append("static initialiser only in AFTER")
    elif ia and not ib:
        fewer = True
    elif ia and ib and ia.calls == ib.calls:
        if ia.shape == ib.shape and ia.targets != ib.targets or ia.immediates != ib.immediates:
            real.append("static initialiser does different work")
        elif ia.shape != ib.shape:
            differing.append(INITIALISER)
    elif ia and ib:
        if ib.calls - ia.calls:
            real.append("static initialiser does more work")
        else:
            fewer = True

    if real:
        return "CHANGED", real[:8]
    cls = "reordered"
    if differing:
        cls = "scheduling"
        why.append("%d functions scheduled differently, e.g. %s" % (len(differing), differing[0]))
    if inlined:
        cls = "inlining"
        why += ["library inlined on one side only: " + n for n in inlined[:6]]

    # Symbols only one side has.
    one_sided = sorted((set(sa) ^ set(sb)) - {INITIALISER})
    private, moved, bad = 0, [], []
    refs_a, refs_b = a.referenced(), b.referenced()
    for name in one_sided:
        in_a = name in sa
        letter = (sa if in_a else sb)[name][0]
        if not strong(letter):
            if name in (refs_b if in_a else refs_a):
                bad.append("one-sided but still referenced: " + name)
            private += 1
            continue
        # External: it must now be defined, unchanged, by another object.
        if in_a:
            where = after.find(name, a.content(name), rel_b)
        else:
            where = before.find(name, b.content(name), rel_a)
        if where is None:
            bad.append("external %s only on one side: %s" % (letter, name))
        else:
            moved.append("%s %s: %s" % ("moved to" if in_a else "moved from", where, name))
    if bad:
        return "CHANGED", bad[:8] + why
    if private:
        cls = worst(cls, "pruned")
        why.append("%d inline/local symbols on one side only, none referenced" % private)
    if moved:
        cls = worst(cls, "moved")
        why += moved[:8]
    if fewer:
        cls = worst(cls, "pruned")
        why.append("static initialiser constructs fewer objects")
    return cls, why


def classify_unpaired(obj, other, side):
    # An object file that exists on one side only: everything it defines with
    # external linkage must be defined, unchanged, somewhere on the other side.
    here = "AFTER" if side == "BEFORE" else "BEFORE"
    syms, fns = obj.symbols(), obj.functions()
    bad, moved, roots = [], [], []
    for name, (letter, _, _) in sorted(syms.items()):
        if strong(letter):
            roots.append(name)
            where = other.find(name, obj.content(name), None)
            if where is None:
                bad.append("external %s not defined in %s: %s" % (letter, side, name))
            else:
                moved.append("%s %s: %s" % ("moved from" if side == "BEFORE" else "moved to", where, name))
    if INITIALISER in fns:
        bad.append("has a static initialiser")
    # A local function is compared as part of whatever reaches it. One that
    # nothing compared reaches, but something refers to, is not compared at
    # all. (Weak template and inline copies are not local: another object
    # has the same one.)
    reached = obj.reached(roots)
    refs = obj.referenced()
    loose = [n for n in fns if syms.get(n, ("?",))[0] == "t" and n not in reached and n != INITIALISER
             and "local " + n not in {t for f in fns.values() for t in f.targets} and n in refs]
    bad += ["local function not compared: " + n for n in loose]
    if bad:
        return "CHANGED", ["only in " + here] + bad[:8]
    return "moved", ["only in " + here] + moved[:8]


def main():
    args, renames = sys.argv[1:], []
    while len(args) > 2 and args[0] == "--rename":
        old, sep, new = args[1].partition("=")
        if not sep or not old:
            sys.exit("--rename wants OLD=NEW")
        renames.append((old, new))
        args = args[2:]
    if len(args) != 2:
        sys.exit(__doc__)
    sys.setrecursionlimit(100000)
    before, after = Tree(Path(args[0]), renames), Tree(Path(args[1]))
    for side, tree in (("BEFORE", before), ("AFTER", after)):
        if not tree.objs:
            print("no object files under %s (%s)" % (side, args[0] if side == "BEFORE" else args[1]))
            return 2
    kinds = Kinds(before, after)

    # Pair objects by path; one that only moved, by file name, then by bytes.
    pairs = [(r, r) for r in after.objs if r in before.objs]
    only_b = [r for r in before.objs if r not in after.objs]
    only_a = [r for r in after.objs if r not in before.objs]
    for r in list(only_a):
        same = [q for q in only_b if q.name == r.name]
        if len(same) == 1 and sum(q.name == r.name for q in only_a) == 1:
            pairs.append((same[0], r))
            only_b.remove(same[0])
            only_a.remove(r)
    for r in list(only_a):
        same = [q for q in only_b if before.objs[q].path.read_bytes() == after.objs[r].path.read_bytes()]
        if same:
            pairs.append((same[0], r))
            only_b.remove(same[0])
            only_a.remove(r)

    rows = []
    for rb, ra in pairs:
        cls, why = classify(before.objs[rb], after.objs[ra], before, after, rb, ra, kinds)
        if rb != ra:
            why = ["was " + str(rb)] + why
            cls = worst(cls, "moved")
        rows.append((str(ra), cls, why))
    for r in only_a:
        rows.append((str(r),) + classify_unpaired(after.objs[r], before, "BEFORE"))
    for r in only_b:
        rows.append((str(r),) + classify_unpaired(before.objs[r], after, "AFTER"))
    rows.sort()

    counts = collections.Counter(r[1] for r in rows)
    for name, cls, why in rows:
        if cls != "bit-identical":
            print("%-13s %s" % (cls, name))
            for w in why:
                print("              " + w[:150])
    print()
    for cls in CLASSES:
        print("%-14s %d" % (cls, counts.get(cls, 0)))
    return 1 if counts.get("CHANGED") else 0


if __name__ == "__main__":
    sys.exit(main())
