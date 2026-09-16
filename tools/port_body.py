#!/usr/bin/env python3
"""
Rewrites ORB-SLAM3's include graph from its bundled Thirdparty/ tree onto the
pinned upstream packages and the vendor_ext wrappers.

Running this is how src/ and include/ were derived from reference/ORB_SLAM3 --
keeping it a script rather than a hand edit means the port stays auditable and
can be replayed against a different upstream pin.

Idempotent: the rewritten forms no longer match the patterns, so a second run
reports nothing.

  ./tools/port_body.py --check    # report what would change, touch nothing
  ./tools/port_body.py
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["src", "include", "Examples"]
SUFFIXES = {".h", ".hpp", ".cc", ".cpp"}

# Matches ORB-SLAM3's include spellings, which vary: `#include "x"`,
# `#include<x>`, `#include "x"` with and without a space after the directive.
INCLUDE_RE = re.compile(r'(#\s*include\s*)([<"])([^">]+)([>"])')

# Where each bundled header lives now.  g2o's type and solver headers all route
# through the compat wrapper, which pulls in the upstream headers and supplies
# the two renamed symbols -- so call sites keep their old spelling.
G2O_COMPAT = "orbslam3r/g2o_ext/compat.hpp"

REMAP = {
    # --- DBoW2 / DLib: now the pinned upstream packages ---------------------
    "Thirdparty/DBoW2/DBoW2/BowVector.h": "DBoW2/BowVector.h",
    "Thirdparty/DBoW2/DBoW2/FeatureVector.h": "DBoW2/FeatureVector.h",
    "Thirdparty/DBoW2/DBoW2/FORB.h": "DBoW2/FORB.h",
    "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h": "DBoW2/TemplatedVocabulary.h",
    "Thirdparty/DBoW2/DUtils/Random.h": "DUtils/Random.h",

    # --- g2o: upstream reorganised types/ and solvers/ into subdirectories --
    "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h": G2O_COMPAT,
    "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h": G2O_COMPAT,
    "Thirdparty/g2o/g2o/types/types_sba.h": G2O_COMPAT,
    "Thirdparty/g2o/g2o/types/sim3.h": G2O_COMPAT,
    "Thirdparty/g2o/g2o/types/se3quat.h": G2O_COMPAT,
    "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h":
        "g2o/solvers/eigen/linear_solver_eigen.h",
    "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h":
        "g2o/solvers/dense/linear_solver_dense.h",

    # --- Sophus: header-only upstream, same layout --------------------------
    "Thirdparty/Sophus/sophus/geometry.hpp": "sophus/geometry.hpp",
    "Thirdparty/Sophus/sophus/sim3.hpp": "sophus/sim3.hpp",
    "Thirdparty/Sophus/sophus/se3.hpp": "sophus/se3.hpp",
    "Thirdparty/Sophus/sophus/so3.hpp": "sophus/so3.hpp",
}

# g2o/core/* keeps its path; only the Thirdparty prefix goes away.
CORE_PREFIX = "Thirdparty/g2o/g2o/"


def remap(path: str):
    if path in REMAP:
        return REMAP[path]
    if path.startswith(CORE_PREFIX):
        return path[len("Thirdparty/g2o/"):]      # -> g2o/core/...
    return None


def rewrite(text: str, counter: Counter) -> str:
    def sub(m):
        directive, open_c, path, close_c = m.groups()
        new = remap(path)
        if new is None:
            return m.group(0)
        counter[f"{path}  ->  {new}"] += 1
        # Everything the port points at is now an installed package or a
        # wrapper on the include path, so angle brackets are the honest form.
        return f"#include <{new}>"
    return INCLUDE_RE.sub(sub, text)


# ORBVocabulary.h names the forked DBoW2 vocabulary directly; the remaster's
# vocabulary is the wrapper subclass that can read ORBvoc.txt.
ORB_VOCABULARY_TYPEDEF = re.compile(
    r"typedef\s+DBoW2::TemplatedVocabulary\s*<\s*DBoW2::FORB::TDescriptor\s*,\s*"
    r"DBoW2::FORB\s*>\s*\n?\s*ORBVocabulary\s*;",
    re.MULTILINE,
)
ORB_VOCABULARY_REPLACEMENT = (
    "// Was DBoW2::TemplatedVocabulary directly, which only ORB-SLAM3's edited\n"
    "// copy could load from ORBvoc.txt.  vendor_ext supplies that format as a\n"
    "// subclass of untouched upstream instead; see docs/WRAPPERS.md.\n"
    "typedef orbslam3r::ORBVocabulary ORBVocabulary;"
)


def normalise_extensions(check: bool):
    """ORB-SLAM3 ships most sources as .cc and a handful as .cpp. Settle on
    .cpp so the tree has one convention."""
    renamed = []
    sources = [f for d in ("src", "Examples") for f in sorted((ROOT / d).rglob("*.cc"))]
    for f in sources:
        dest = f.with_suffix(".cpp")
        renamed.append((f.relative_to(ROOT), dest.relative_to(ROOT)))
        if not check:
            f.rename(dest)
    return renamed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    renamed = normalise_extensions(args.check)
    if renamed:
        verb = "would rename" if args.check else "renamed"
        print(f"{verb} {len(renamed)} sources from .cc to .cpp\n")

    counter = Counter()
    touched = []

    for target in TARGETS:
        base = ROOT / target
        if not base.is_dir():
            sys.exit(f"missing directory: {base}")
        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.suffix not in SUFFIXES:
                continue
            original = f.read_text(encoding="utf-8", errors="replace")
            text = rewrite(original, counter)

            if f.name == "ORBVocabulary.h":
                text, n = ORB_VOCABULARY_TYPEDEF.subn(
                    ORB_VOCABULARY_REPLACEMENT, text)
                if n:
                    counter["ORBVocabulary -> orbslam3r::ORBVocabulary"] += n
                    text = text.replace(
                        "#include <DBoW2/TemplatedVocabulary.h>",
                        "#include <orbslam3r/dbow2_ext/orb_vocabulary.hpp>")

            if text != original:
                touched.append(f.relative_to(ROOT))
                if not args.check:
                    f.write_text(text, encoding="utf-8")

    if not counter:
        print("nothing to rewrite (already ported)")
        return 0

    width = max(len(k) for k in counter)
    print(f"{'REWRITE'.ljust(width)}  COUNT")
    print(f"{'-' * width}  -----")
    for k, v in sorted(counter.items()):
        print(f"{k.ljust(width)}  {v}")
    print()
    verb = "would touch" if args.check else "touched"
    print(f"{verb} {len(touched)} files, {sum(counter.values())} rewrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
