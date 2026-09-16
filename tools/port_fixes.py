#!/usr/bin/env python3
"""
Stage three of the port: targeted source fixes that neither the include rewrite
(port_body.py) nor the std:: qualification (qualify_std.py) can express.

Each fix states what it changes and why.  Keeping them here rather than editing
src/ by hand means the whole port replays from the pristine reference checkout:

    rm -rf src include && cp -R reference/ORB_SLAM3/{src,include} .
    ./tools/port_body.py && ./tools/qualify_std.py && ./tools/port_fixes.py

  ./tools/port_fixes.py --check
  ./tools/port_fixes.py
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
def fix_g2o_solver_ownership(text, path):
    """Route g2o solver construction through vendor_ext's factory.

    ORB-SLAM3 builds optimizers the way g2o worked in 2016, with raw owning
    pointers threaded through three constructors:

        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
        linearSolver = new g2o::LinearSolverEigen<...::PoseMatrixType>();
        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    Current g2o takes std::unique_ptr at both levels, so every one of these is a
    compile error -- 33 of them in this file.  MakeLevenberg/MakeGaussNewton
    collapse the whole preamble into one call and keep the ownership convention
    in a single header.

    A side effect worth noting: where the original creates one solver_ptr and
    then branches to build two algorithms from it (BlockSolverX in the bLarge
    case), only one algorithm is ever used and the other silently leaks the
    block solver.  Building the algorithm inside each branch removes that.
    """
    if path.name != "Optimizer.cpp":
        return text, 0

    lines = text.splitlines(keepends=True)
    out = []
    block_solver = None      # e.g. "g2o::BlockSolver_6_3"
    kind = "kEigen"
    n = 0

    decl_re = re.compile(
        r"^\s*(g2o::BlockSolver\w*)::LinearSolverType\s*\*\s*linearSolver\s*;")
    decl_assign_re = re.compile(
        r"^\s*(g2o::BlockSolver\w*)::LinearSolverType\s*\*\s*linearSolver\s*=")
    assign_re = re.compile(
        r"^\s*linearSolver\s*=\s*new\s+g2o::LinearSolver(Eigen|Dense)\s*<")
    inline_new_re = re.compile(
        r"^\s*new\s+g2o::LinearSolver(Eigen|Dense)\s*<")
    ptr_re = re.compile(
        r"^\s*(g2o::BlockSolver\w*)\s*\*\s*\w+\s*=\s*new\s+g2o::BlockSolver\w*\s*\(\s*linearSolver\s*\)\s*;")
    algo_re = re.compile(
        r"(g2o::OptimizationAlgorithm(Levenberg|GaussNewton))\s*\*\s*(\w+)\s*="
        r"\s*new\s+g2o::OptimizationAlgorithm(?:Levenberg|GaussNewton)\s*\(\s*\w+\s*\)\s*;")

    for line in lines:
        m = decl_re.match(line) or decl_assign_re.match(line)
        if m:
            block_solver = m.group(1)
            # A declaration that also assigns carries its kind on this line or
            # the next; assign_re/inline_new_re pick it up either way.
            km = re.search(r"LinearSolver(Eigen|Dense)", line)
            if km:
                kind = "kEigen" if km.group(1) == "Eigen" else "kDense"
            n += 1
            continue
        m = assign_re.match(line) or inline_new_re.match(line)
        if m:
            kind = "kEigen" if m.group(1) == "Eigen" else "kDense"
            n += 1
            continue
        m = ptr_re.match(line)
        if m:
            block_solver = m.group(1)
            n += 1
            continue
        m = algo_re.search(line)
        if m and block_solver:
            algo = "MakeLevenberg" if m.group(2) == "Levenberg" else "MakeGaussNewton"
            indent = re.match(r"^\s*", line).group(0)
            replacement = (
                f"{indent}auto* {m.group(3)} = orbslam3r::g2o_ext::{algo}<"
                f"{block_solver}, orbslam3r::g2o_ext::LinearSolver::{kind}>();\n")
            out.append(replacement)
            n += 1
            continue
        out.append(line)

    text = "".join(out)

    # Collapse the blank lines the removed declarations left behind.
    text = re.sub(r"\n{3,}", "\n\n", text)

    if n and "g2o_ext/solver_factory.hpp" not in text:
        text = text.replace(
            "#include <orbslam3r/g2o_ext/compat.hpp>",
            "#include <orbslam3r/g2o_ext/compat.hpp>\n"
            "#include <orbslam3r/g2o_ext/solver_factory.hpp>", 1)
    return text, n


# ---------------------------------------------------------------------------
def fix_full_ba_generation_counter(text, path):
    """`mnFullBAIdx` is incremented and compared, but declared bool.

    Three call sites abort a running global bundle adjustment with

        mbStopGBA = true;
        mnFullBAIdx++;

    and RunGlobalBundleAdjustment guards its map update with

        int idx = mnFullBAIdx;
        { lock(mMutexGBA); if (idx != mnFullBAIdx) return; ... }

    With `bool mnFullBAIdx;` the first `++` sets it to true and every later one
    is a no-op, so after the first abort in a session `idx != mnFullBAIdx` can
    never be true again and that guard is permanently dead. ORB-SLAM2 declared
    the same member `int`.

    C++17 removed operator++ on bool, which is the only reason this surfaced.

    SCOPE: only the type is changed here. ORB-SLAM3 also reads the snapshot
    *after* the optimisation returns, where ORB-SLAM2 read it before, so even as
    an int this guard now only covers the gap between that read and acquiring
    mMutexGBA. Moving the snapshot back would change runtime behaviour rather
    than fix a compile error, and the primary abort handling is mbStopGBA, which
    is checked separately a few lines below.
    """
    if path.stem != "LoopClosing" or path.suffix not in (".h", ".hpp"):
        return text, 0
    new, n = re.subn(
        r"^(\s*)bool(\s+mnFullBAIdx\s*;)",
        r"\1// Counter, not a flag: incremented on every GBA abort and compared\n"
        r"\1// with != in RunGlobalBundleAdjustment. As a bool it saturates at\n"
        r"\1// true and that comparison stops working. See tools/port_fixes.py.\n"
        r"\1int\2",
        text, flags=re.MULTILINE)
    return new, n


# ---------------------------------------------------------------------------
def add_dbow2_serialization(text, path):
    """KeyFrame archives DBoW2::BowVector and FeatureVector.

    ORB-SLAM3 made that work by adding an intrusive serialize() member to its
    forked copies of both classes.  vendor_ext supplies the same capability
    non-intrusively, but the overloads have to be visible where the archive is
    instantiated.
    """
    if path.stem != "KeyFrame" or path.suffix not in (".h", ".hpp"):
        return text, 0
    if "dbow2_ext/serialization.hpp" in text:
        return text, 0
    anchor = "#include <boost/serialization/map.hpp>"
    if anchor not in text:
        return text, 0
    return text.replace(
        anchor,
        anchor + "\n\n"
        "// Non-intrusive Boost support for DBoW2::BowVector / FeatureVector,\n"
        "// replacing the intrusive members ORB-SLAM3 added to its DBoW2 fork.\n"
        "#include <orbslam3r/dbow2_ext/serialization.hpp>", 1), 1


# ---------------------------------------------------------------------------
COMPILEDWITHC11_RE = re.compile(
    r"^[ \t]*#[ \t]*ifdef[ \t]+COMPILEDWITHC11[ \t]*\r?\n"
    r"(?P<cxx11>.*?)"
    r"^[ \t]*#[ \t]*else[ \t]*\r?\n"
    r".*?"
    r"^[ \t]*#[ \t]*endif[ \t]*\r?\n",
    re.MULTILINE | re.DOTALL,
)


def drop_compiledwithc11_guards(text, path):
    """Collapse `#ifdef COMPILEDWITHC11` to its C++11 branch.

    Upstream guards every timing call like this:

        #ifdef COMPILEDWITHC11
            auto t1 = std::chrono::steady_clock::now();
        #else
            auto t1 = std::chrono::monotonic_clock::now();
        #endif

    and relies on its CMake defining COMPILEDWITHC11.  The #else branch names
    std::chrono::monotonic_clock, which is a pre-standard name that never
    existed in C++11 or later -- so that branch cannot compile on any current
    toolchain, and the macro only ever hid it.  This project is C++17, so the
    guard is deleted and the live branch kept.
    """
    if not COMPILEDWITHC11_RE.search(text):
        return text, 0
    return COMPILEDWITHC11_RE.subn(lambda m: m.group("cxx11"), text)


# ---------------------------------------------------------------------------
def make_viewer_switchable(text, path):
    """Let ORBSLAM3R_VIEWER=0 force the Pangolin viewer off.

    Upstream's examples disagree about the flag they pass: mono_euroc asks for
    no viewer, stereo_euroc asks for one, and stereo_inertial_euroc asks for
    none again.  Inside a container the viewer runs on llvmpipe over Xvfb, where
    it drags a batch run to a crawl -- stereo_euroc on EuRoC MH01 made no
    measurable progress in twelve minutes with it on, versus three and a half
    for the whole sequence with it off.

    Upstream clearly hit this too; the line below the check is their own
    commented-out `if(false) // TODO`.

    The default is unchanged: with the variable unset, the caller's flag wins.
    """
    if path.name != "System.cpp":
        return text, 0
    old = "    //Initialize the Viewer thread and launch\n    if(bUseViewer)\n"
    if old not in text:
        return text, 0
    new = (
        "    //Initialize the Viewer thread and launch\n"
        "    // ORBSLAM3R_VIEWER=0 forces the viewer off whatever the caller\n"
        "    // asked for, so a headless batch run has one switch. Unset, the\n"
        "    // caller's flag wins and behaviour is unchanged.\n"
        "    const char* viewerEnv = std::getenv(\"ORBSLAM3R_VIEWER\");\n"
        "    const bool bViewerEnabled =\n"
        "        bUseViewer && !(viewerEnv && std::string(viewerEnv) == \"0\");\n"
        "    if(bViewerEnabled)\n"
    )
    text = text.replace(old, new, 1)
    if "#include <cstdlib>" not in text:
        text = text.replace("#include <iostream>",
                            "#include <cstdlib>\n#include <iostream>", 1)
    return text, 1


FIXES = [
    ("COMPILEDWITHC11 guards", drop_compiledwithc11_guards),
    ("viewer env switch", make_viewer_switchable),
    ("g2o solver ownership", fix_g2o_solver_ownership),
    ("mnFullBAIdx bool -> int", fix_full_ba_generation_counter),
    ("DBoW2 serialization include", add_dbow2_serialization),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    totals = {name: 0 for name, _ in FIXES}
    touched = set()

    for target in ("include", "src", "Examples"):
        for f in sorted((ROOT / target).rglob("*")):
            if not f.is_file() or f.suffix not in {".h", ".hpp", ".cpp"}:
                continue
            original = f.read_text(encoding="utf-8", errors="replace")
            text = original
            for name, fn in FIXES:
                text, n = fn(text, f)
                totals[name] += n
            if text != original:
                touched.add(f.relative_to(ROOT))
                if not args.check:
                    f.write_text(text, encoding="utf-8")

    for name, n in totals.items():
        print(f"  {name:<32} {n}")
    print(f"\nfiles touched: {len(touched)}")
    for t in sorted(touched):
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
