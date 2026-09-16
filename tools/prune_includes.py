#!/usr/bin/env python3
"""
Removes header includes that are redundant with a forward declaration in the
same header.

WHY THESE EXIST
  ORB-SLAM3's headers both #include and forward-declare the same classes -- 48
  such pairs across 14 headers when this was written. The two cancel out: the
  include creates a cycle, and the forward declaration is what lets the cycle
  compile at all, with include guards deciding the outcome by whichever header
  is reached first. So the forward declarations are not a compile-time
  optimisation, they are scar tissue.

  Where a header only ever names the class through a pointer or reference, the
  forward declaration alone is correct and the include is what should go. The
  definition then belongs in the .cpp that actually dereferences it.

WHAT THIS DOES NOT DECIDE
  Whether the header truly needs only an incomplete type. It cannot: base
  classes, by-value members and inline bodies all need the definition. The
  compiler decides -- remove, build, and put back whatever fails. Run with
  --only to work through a subset.

  ./tools/prune_includes.py --list
  ./tools/prune_includes.py [--only Header1,Header2]
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = ROOT / "include"


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def redundant_pairs():
    own = {h.stem for h in INCLUDE.rglob("*.hpp")}
    pairs = {}
    for h in sorted(INCLUDE.rglob("*.hpp")):
        bare = strip_comments(h.read_text(errors="replace"))
        fwd = set(re.findall(r"^\s*class\s+([A-Z]\w*)\s*;", bare, re.M)) & own
        included = {m.split("/")[-1]
                    for m in re.findall(r'#include\s*[<"]([A-Za-z0-9_/]+)\.hpp[>"]', bare)} & own
        both = sorted((fwd & included) - {h.stem})
        if both:
            pairs[h] = both
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="report and change nothing")
    ap.add_argument("--only", help="comma-separated header stems to act on")
    args = ap.parse_args()

    pairs = redundant_pairs()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        pairs = {h: v for h, v in pairs.items() if h.stem in wanted}

    if not pairs:
        print("no redundant include/forward-declaration pairs")
        return 0

    total = 0
    for header, classes in pairs.items():
        text = header.read_text(errors="replace")
        removed = []
        for cls in classes:
            # Only the "Name.hpp" spelling appears in this tree.
            line = re.compile(r'^[ \t]*#include[ \t]*"[^"]*\b' + cls + r'\.hpp"[ \t]*\r?\n',
                              re.M)
            text, n = line.subn("", text)
            if n:
                removed.append(cls)
                total += n
        if removed and not args.list:
            header.write_text(text)
        print(f"{header.relative_to(ROOT)}: {', '.join(removed)}")

    verb = "would remove" if args.list else "removed"
    print(f"\n{verb} {total} includes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
