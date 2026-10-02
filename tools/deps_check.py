#!/usr/bin/env python3
"""
Holds every #include to what the build says its part may use.

  ./tools/deps_check.py [--graph build/part_graph.txt]            the report
  ./tools/deps_check.py --check [--graph build/part_graph.txt]    and exit 1 on a breach

src/CMakeLists.txt puts the library together from parts, and for each part it
names the parts and the packages that part uses. CMake does not hold a part to
that: every header of the project is under one include/ and every package
under one prefix, so anything can include anything and still compile. This
does. It reads the parts as CMake wrote them out (part_graph.txt, one line per
part: name|layers|sources|parts used|packages) and reports

  a project header   of a part that the including part neither is nor names
  a package header   of a package the including part does not name
  a cycle            among the parts

A header belongs to the part that has the source of the same name in the same
folder, and otherwise to the part that has the folder; a header at the root of
include/ belongs to the part that has System.cpp.

What is named is what is used directly. A part that reaches a package through
another part's headers has not named it, and says so here when one of its own
files includes it.

WAIVED lists the includes that are known to break this and are to be undone;
one that no longer occurs has to be taken off the list, or this fails.
"""
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The first component of a package's include paths -> how the build names it.
PACKAGE_OF = {
    "Eigen": "Eigen3::Eigen",
    "opencv2": "opencv",
    "boost": "Boost::serialization",
    "sophus": "Sophus",
    "openssl": "OpenSSL::Crypto",
    "pangolin": "pangolin",
    "GL": "pangolin",
    "g2o": "orbslam3r::g2o_ext",
    "DBoW2": "orbslam3r::dbow2_ext",
    "DUtils": "orbslam3r::dbow2_ext",
}

# (file, include) pairs that break the rule today. None: Tracking held the
# viewer's two drawers and the System, and System made the viewer; all four
# are ports now (tracking/ViewPorts.hpp, common/ThreadPorts.hpp,
# ViewerPort.hpp).
WAIVED = set()


def package_names(words):
    """What a part's PACKAGES line names, as the names PACKAGE_OF uses."""
    out = set()
    for w in words:
        if w.startswith("opencv_"):
            out.add("opencv")
        elif w.startswith("pango"):
            out.add("pangolin")
        else:
            out.add(w)
    return out


def read_graph(path):
    parts = collections.OrderedDict()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, layers, files, uses, packages = line.split("|")
        parts[name] = {
            "layers": layers.split(),
            "files": files.split(),
            "uses": uses.split(),
            "packages": package_names(packages.split()),
        }
    return parts


def cycle_in(parts):
    state = {}

    def visit(n, path):
        if state.get(n) == 1:
            return path[path.index(n):] + [n]
        if state.get(n) == 2:
            return None
        state[n] = 1
        for m in parts[n]["uses"]:
            if m in parts:
                found = visit(m, path + [n])
                if found:
                    return found
        state[n] = 2
        return None

    for n in parts:
        found = visit(n, [])
        if found:
            return found
    return None


def main():
    check = "--check" in sys.argv
    graph = ROOT / "build" / "part_graph.txt"
    if "--graph" in sys.argv:
        graph = pathlib.Path(sys.argv[sys.argv.index("--graph") + 1])
    if not graph.exists():
        print(f"{graph} not found: configure the build first", file=sys.stderr)
        return 2
    parts = read_graph(graph)

    source_owner, layer_owner, top = {}, {}, None
    for name, p in parts.items():
        for f in p["files"]:
            source_owner[f] = name
            if f == "System.cpp":
                top = name
        for layer in p["layers"]:
            layer_owner[layer] = name

    def owner_of_header(rel):  # rel: path under include/
        q = pathlib.PurePosixPath(rel)
        if len(q.parts) == 1:
            return top
        twin = str(q.with_suffix(".cpp"))
        return source_owner.get(twin) or layer_owner.get(q.parts[0])

    files = []
    for f in sorted((ROOT / "src").rglob("*.cpp")):
        rel = str(f.relative_to(ROOT / "src"))
        files.append((f, source_owner.get(rel)))
    for f in sorted((ROOT / "include").rglob("*.hpp")):
        files.append((f, owner_of_header(str(f.relative_to(ROOT / "include")))))

    breaches, waived_seen, unowned = [], set(), []
    edges = collections.defaultdict(collections.Counter)
    for f, part in files:
        shown = str(f.relative_to(ROOT))
        if part is None:
            unowned.append(shown)
            continue
        for m in re.finditer(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', f.read_text(errors="replace"), re.M):
            inc = m.group(1)
            if (ROOT / "include" / inc).exists():
                other = owner_of_header(inc)
                if other == part:
                    continue
                edges[part][other] += 1
                if other in parts[part]["uses"]:
                    continue
                what = f"part '{other}', which '{part}' does not use"
            else:
                first = inc.split("/")[0]
                if first == "orbslam3r" and inc.count("/") >= 2:
                    package = "orbslam3r::" + inc.split("/")[1]  # a vendor_ext wrapper
                elif "/" in inc and first in PACKAGE_OF:
                    package = PACKAGE_OF[first]
                else:
                    continue  # the standard library and the system
                if package in parts[part]["packages"]:
                    continue
                what = f"package '{package}', which '{part}' does not name"
            if (shown, inc) in WAIVED:
                waived_seen.add((shown, inc))
            else:
                breaches.append(f"{shown}: \"{inc}\" is of {what}")

    cycle = cycle_in(parts)
    stale = sorted(WAIVED - waived_seen)

    width = max(len(n) for n in parts)
    for name, p in parts.items():
        used = ", ".join(f"{o} ({edges[name][o]})" if edges[name][o] else f"{o} (0)" for o in p["uses"]) or "-"
        print(f"{name:{width}}  {len(p['files']):2d} sources  uses {used}")
    print(f"breaches   {len(breaches)}")
    for b in breaches:
        print("    " + b)
    print(f"waived     {len(waived_seen)}")
    for w in sorted(waived_seen):
        print(f"    {w[0]}: \"{w[1]}\"")
    if stale:
        print(f"stale waivers  {len(stale)}  (no longer occur: take them off the list)")
        for w in stale:
            print(f"    {w[0]}: \"{w[1]}\"")
    if unowned:
        print(f"files of no part  {len(unowned)}")
        for u in unowned:
            print("    " + u)
    print("cycle among parts  " + (" -> ".join(cycle) if cycle else "none"))

    broken = bool(breaches) or bool(stale) or bool(unowned) or bool(cycle)
    if check and broken:
        print("\nFAIL")
    return 1 if (check and broken) else 0


if __name__ == "__main__":
    sys.exit(main())
