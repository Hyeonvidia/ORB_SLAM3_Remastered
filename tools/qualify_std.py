#!/usr/bin/env python3
"""
Replaces ORB-SLAM3's reliance on `using namespace std` with explicit std::
qualification, drops the `using namespace std;` lines, and adds the standard
headers the code had been getting transitively.

WHY THIS IS NEEDED AT ALL
  ORB-SLAM3's own headers never say `using namespace std;` -- they inherited it.
  Its forked copy of DBoW2 puts one at global scope in
  Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h:36, and ORBVocabulary.h drags
  that into most of the system.  Upstream DBoW2 has no such line, so the moment
  the remaster switched to the pristine upstream package, ~2,300 unqualified
  names stopped resolving.  Several headers were also relying on it for their
  <string> / <vector> includes.

  Reintroducing the `using` would be one line.  Qualifying is the better answer:
  namespace pollution stops leaking through headers into everything downstream.

HOW NAMES ARE MATCHED
  A blanket "any word in this list" rule is unsafe -- `int count = 0;` would
  become `int std::count = 0;`.  So names are matched by the shape of their use:

    TEMPLATES  qualified only before `<`     vector<  map<  numeric_limits<
    FUNCTIONS  qualified only before `(`     sort(  min(  to_string(
    NAMESPACES qualified only before `::`    chrono::  this_thread::
    TYPES      qualified anywhere            string  mutex  ofstream
    OBJECTS    qualified anywhere            cout  cerr  endl

  Only TYPES and OBJECTS are matched positionally, and those names are
  distinctive enough not to collide with ORB-SLAM3's own identifiers.

  Comments, string/char literals and preprocessor directives are masked out
  first, so prose like "list of keyframes" and `#include <vector>` are safe.

  ./tools/qualify_std.py --check
  ./tools/qualify_std.py
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["include", "src", "Examples"]
SUFFIXES = {".h", ".hpp", ".cc", ".cpp"}

TEMPLATES = [
    "vector", "map", "list", "set", "pair", "array", "tuple", "deque",
    "unordered_map", "unordered_set", "multimap", "multiset",
    "shared_ptr", "unique_ptr", "weak_ptr",
    "numeric_limits", "unique_lock", "lock_guard", "function",
    "make_shared", "make_unique", "get",
]
FUNCTIONS = [
    "make_pair", "sort", "stable_sort", "min", "max", "min_element",
    "max_element", "swap", "accumulate", "getline",
    "to_string", "stoi", "stof", "stod", "setprecision", "setw", "setfill",
    "sqrt", "pow", "abs", "fabs", "isfinite", "isnan", "round", "floor", "ceil",
    "ref", "cref",
]
NAMESPACES = ["chrono", "this_thread", "ios_base", "ios"]
TYPES = [
    "string", "ofstream", "ifstream", "fstream", "stringstream",
    "istringstream", "ostringstream", "ostream", "istream",
    "mutex", "recursive_mutex", "thread", "condition_variable",
    "exception", "runtime_error", "invalid_argument", "out_of_range",
    "size_t",
]
OBJECTS = ["cout", "cerr", "cin", "endl", "flush", "fixed", "scientific"]

# `size_t` resolves without std:: through <cstddef>; leave it alone.
TYPES.remove("size_t")

# NOTE: `find` and `count` are deliberately absent from FUNCTIONS.  ORB-SLAM3
# declares Sim3Solver::find(...) and uses `count` as a local, so matching them
# before "(" would rewrite the declarations themselves.  Any genuine
# std::find/std::count call is qualified by hand in port_fixes.py.

HEADER_FOR = {
    "vector": "vector", "map": "map", "multimap": "map", "list": "list",
    "set": "set", "multiset": "set", "deque": "deque", "array": "array",
    "tuple": "tuple", "get": "tuple",
    "unordered_map": "unordered_map", "unordered_set": "unordered_set",
    "pair": "utility", "make_pair": "utility", "swap": "utility",
    "shared_ptr": "memory", "unique_ptr": "memory", "weak_ptr": "memory",
    "make_shared": "memory", "make_unique": "memory",
    "string": "string", "to_string": "string", "stoi": "string",
    "stof": "string", "stod": "string",
    "cout": "iostream", "cerr": "iostream", "cin": "iostream",
    "endl": "iostream", "flush": "iostream",
    "ostream": "ostream", "istream": "istream",
    "ofstream": "fstream", "ifstream": "fstream", "fstream": "fstream",
    "stringstream": "sstream", "istringstream": "sstream",
    "ostringstream": "sstream", "getline": "string",
    "setprecision": "iomanip", "setw": "iomanip", "setfill": "iomanip",
    "fixed": "iomanip",
    "scientific": "iomanip",
    "mutex": "mutex", "recursive_mutex": "mutex", "unique_lock": "mutex",
    "lock_guard": "mutex",
    "thread": "thread", "this_thread": "thread",
    "condition_variable": "condition_variable",
    "chrono": "chrono",
    "sort": "algorithm", "stable_sort": "algorithm", "min": "algorithm",
    "max": "algorithm", "min_element": "algorithm", "max_element": "algorithm",
    "accumulate": "numeric", "numeric_limits": "limits",
    "sqrt": "cmath", "pow": "cmath", "abs": "cmath", "fabs": "cmath",
    "isfinite": "cmath", "isnan": "cmath", "round": "cmath",
    "floor": "cmath", "ceil": "cmath",
    "function": "functional", "ref": "functional", "cref": "functional",
    "ios_base": "ios", "ios": "ios",
    "exception": "stdexcept", "runtime_error": "stdexcept",
    "invalid_argument": "stdexcept", "out_of_range": "stdexcept",
}

# Not already part of a qualified-id or a member access: excludes std::vector,
# frame.map, kf->set and mSetVertices.
LEAD = r"(?<![\w:.>])"

PATTERNS = [
    re.compile(LEAD + r"(" + "|".join(TEMPLATES) + r")\b(?=\s*<)"),
    re.compile(LEAD + r"(" + "|".join(FUNCTIONS) + r")\b(?=\s*\()"),
    re.compile(LEAD + r"(" + "|".join(NAMESPACES) + r")\b(?=\s*::)"),
    re.compile(LEAD + r"(" + "|".join(TYPES) + r")\b"),
    re.compile(LEAD + r"(" + "|".join(OBJECTS) + r")\b(?!\s*::)"),
]

USING_RE = re.compile(r"^[ \t]*using namespace std\s*;[ \t]*\r?\n", re.MULTILINE)
INCLUDE_RE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]+)[>\"]",
                        re.MULTILINE)


def mask_regions(text):
    """Blank out comments, string/char literals and preprocessor directives,
    preserving length so offsets still line up with the original."""
    out = list(text)
    i, n = 0, len(text)
    at_line_start = True
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if at_line_start and c == "#":
            while i < n:
                if text[i] == "\\" and i + 1 < n and text[i + 1] == "\n":
                    out[i] = " "
                    i += 2
                    continue
                if text[i] == "\n":
                    break
                out[i] = " "
                i += 1
            continue
        if c == "\n":
            at_line_start = True
            i += 1
            continue
        if not c.isspace():
            at_line_start = False

        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and nxt == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
        elif c in "\"'":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    out[i] = " "
                    if i + 1 < n:
                        out[i + 1] = " "
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                if text[i] != "\n":
                    out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


SCOPE_SPACING_RE = re.compile(r"::[ \t]+(?=[A-Za-z_])")


def collapse_scope_spacing(text, counter):
    """Turn `std:: vector` into `std::vector`.

    ORB-SLAM3 writes the scope operator with a trailing space in a few places.
    Python's fixed-width lookbehind cannot see past that space, so `vector`
    would look unqualified and become `std:: std::vector`.  Normalising first
    is simpler than trying to match around it, and is a readability fix anyway.
    """
    masked = mask_regions(text)
    pieces, last = [], 0
    for m in SCOPE_SPACING_RE.finditer(masked):
        pieces.append(text[last:m.start()])
        pieces.append("::")
        counter["(collapsed `:: ` spacing)"] += 1
        last = m.end()
    pieces.append(text[last:])
    return "".join(pieces)


# Finds names already written as std::X, so headers the code was getting
# transitively still get included even where nothing needed qualifying.
ALREADY_QUALIFIED_RE = re.compile(r"\bstd::([A-Za-z_]\w*)")


def qualify(text, counter):
    """Apply every pattern, re-masking between passes so a name qualified by an
    earlier pass is not matched again by a later one."""
    used = set()
    for pattern in PATTERNS:
        masked = mask_regions(text)
        pieces, last = [], 0
        for m in pattern.finditer(masked):
            pieces.append(text[last:m.start()])
            pieces.append("std::" + m.group(1))
            counter[m.group(1)] += 1
            used.add(m.group(1))
            last = m.end()
        pieces.append(text[last:])
        text = "".join(pieces)
    return text, used


def add_includes(text, used, counter):
    """Ensure every std facility the file now names has its header included."""
    needed = {HEADER_FOR[n] for n in used if n in HEADER_FOR}
    present = set(INCLUDE_RE.findall(text))
    missing = sorted(needed - present)
    if not missing:
        return text

    lines = text.splitlines(keepends=True)
    last_include = None
    for idx, line in enumerate(lines):
        if INCLUDE_RE.match(line):
            last_include = idx
    if last_include is None:
        # No includes yet: land just after the include guard, or at the top.
        for idx, line in enumerate(lines):
            if re.match(r"^[ \t]*#[ \t]*define\b", line):
                last_include = idx
                break
        else:
            last_include = -1

    block = "".join(f"#include <{h}>\n" for h in missing)
    lines.insert(last_include + 1, "\n" + block)
    for h in missing:
        counter[f"#include <{h}>"] += 1
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    names = Counter()
    includes = Counter()
    removed = 0
    touched = []

    for target in TARGETS:
        for f in sorted((ROOT / target).rglob("*")):
            if not f.is_file() or f.suffix not in SUFFIXES:
                continue
            original = f.read_text(encoding="utf-8", errors="replace")
            text, n = USING_RE.subn("", original)
            removed += n
            text = collapse_scope_spacing(text, names)
            text, _ = qualify(text, names)
            # Include decisions look at every std:: name in the finished file,
            # not only the ones this pass qualified: ORB-SLAM3 already wrote
            # std::cout in a few headers while relying on the leaked
            # <iostream> to declare it.
            used = set(ALREADY_QUALIFIED_RE.findall(text))
            text = add_includes(text, used, includes)
            if text != original:
                touched.append(f.relative_to(ROOT))
                if not args.check:
                    f.write_text(text, encoding="utf-8")

    print(f"removed `using namespace std;` : {removed}")
    print(f"qualified names               : {sum(names.values())}")
    print(f"standard headers added        : {sum(includes.values())}")
    print(f"files touched                 : {len(touched)}\n")
    for name, n in names.most_common(12):
        print(f"  {name:<18} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
