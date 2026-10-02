#!/usr/bin/env python3
"""
Says which classes know about each other, and which of that is a cycle.

  ./tools/cycles.py            the report
  ./tools/cycles.py --check    the report, and exit 1 if a hard rule is broken

A forward declaration is not a defect. `class Settings;` in a header that only
passes a Settings* along is what keeps the header's includes short, and most of
the forward declarations in include/ are that. What is worth tracking is
different, and this measures it:

  include cycles     headers that include each other, directly or not. A build
                     problem, and a hard rule: there must be none.
  knowledge cycles   groups of classes each of which can reach every other
                     through #include or forward declaration -- they cannot be
                     understood, tested or replaced one at a time. The report
                     lists each group; the aim is for the groups to shrink and
                     for no new one to appear.
  upward includes    an #include from a lower layer to a higher one, in the
                     order below. Counted for headers and for sources.
  dead declarations  a forward declaration of a name the header never uses.
                     A hard rule: none.

Two groups exist today and are not the same kind of thing. KeyFrame, MapPoint
and Map know each other because a covisibility graph is mutual by nature; that
one is to be kept inside atlas/ with one owner, not broken. Tracking, Local
Mapping, Loop Closing, System and the viewer know each other because they call
each other directly, which interfaces owned by the lower layer can undo.
"""
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The viewer is last: it draws what the layers below hold and its buttons call
# System, which knows it only through a port.
ORDER = ["common", "camera", "features", "atlas", "tracking", "optimization",
         "local_mapping", "loop_closing", "System", "viewer"]
RANK = {layer: i for i, layer in enumerate(ORDER)}


def layer_of(path, base):
    parts = path.relative_to(ROOT / base).parts
    return parts[0] if len(parts) > 1 else "System"


def components(graph):
    """Strongly connected components with more than one node (Tarjan, iterative)."""
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]
    for start in graph:
        if start in index:
            continue
        work = [(start, iter(graph[start]))]
        index[start] = low[start] = counter[0]; counter[0] += 1
        stack.append(start); on.add(start)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in graph:
                    continue
                if w not in index:
                    index[w] = low[w] = counter[0]; counter[0] += 1
                    stack.append(w); on.add(w)
                    work.append((w, iter(graph[w])))
                    advanced = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop(); on.discard(w); comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return sorted(out, key=len, reverse=True)


def main():
    check = "--check" in sys.argv
    headers = sorted((ROOT / "include").rglob("*.hpp"))
    stems = {h.stem for h in headers}
    text = {h.stem: h.read_text(errors="replace") for h in headers}
    fwd, inc = {}, {}
    for h in headers:
        s = text[h.stem]
        fwd[h.stem] = sorted({m.group(2) for m in re.finditer(r"^\s*(class|struct)\s+(\w+)\s*;", s, re.M)})
        inc[h.stem] = sorted({pathlib.PurePosixPath(m.group(1)).stem
                              for m in re.finditer(r'#include\s*"([^"]+)"', s)} & stems)

    include_cycles = components({n: set(inc[n]) for n in stems})
    knowledge = {n: (set(fwd[n]) & stems | set(inc[n])) - {n} for n in stems}
    groups = components(knowledge)
    member = {n: i for i, g in enumerate(groups) for n in g}

    dead, inside, one_way = [], collections.Counter(), 0
    for h in headers:
        n = h.stem
        for f in fwd[n]:
            body = re.sub(r"^\s*(class|struct)\s+" + f + r"\s*;", "", text[n], flags=re.M)
            if not re.search(r"\b" + f + r"\b", body):
                dead.append(f"{h.relative_to(ROOT)}: {f}")
            elif n in member and member.get(f) == member[n]:
                inside[member[n]] += 1
            else:
                one_way += 1

    upward = collections.Counter()
    for base in ("include", "src"):
        for p in (ROOT / base).rglob("*"):
            if p.suffix not in (".hpp", ".cpp"):
                continue
            src = layer_of(p, base)
            for m in re.finditer(r'#include\s*"([^"]+)"', p.read_text(errors="replace")):
                q = pathlib.PurePosixPath(m.group(1))
                dst = q.parts[0] if len(q.parts) > 1 and q.parts[0] in RANK else ("System" if q.name == "System.hpp" else None)
                if dst and src in RANK and RANK[dst] > RANK[src]:
                    upward[p.suffix] += 1

    total = sum(len(v) for v in fwd.values())
    print(f"forward declarations        {total}")
    print(f"  one-way, no cycle         {one_way}")
    for i, g in enumerate(groups):
        print(f"  inside knowledge cycle {i + 1}  {inside[i]}")
    print(f"  dead                      {len(dead)}")
    for d in dead:
        print(f"      {d}")
    print(f"include cycles              {len(include_cycles)}")
    for g in include_cycles:
        print("      " + ", ".join(g))
    print(f"knowledge cycles            {len(groups)}")
    for i, g in enumerate(groups):
        print(f"  {i + 1}: {len(g)} classes: " + ", ".join(g))
    print(f"upward includes             {upward['.hpp']} in headers, {upward['.cpp']} in sources")

    broken = bool(include_cycles) or bool(dead)
    if check and broken:
        print("\nFAIL: include cycles and dead forward declarations are not allowed")
    return 1 if (check and broken) else 0


if __name__ == "__main__":
    sys.exit(main())
