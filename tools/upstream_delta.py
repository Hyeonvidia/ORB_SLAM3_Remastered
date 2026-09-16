#!/usr/bin/env python3
"""
Extracts the MINIMAL delta between a file ORB-SLAM3 vendored and its closest
ancestor anywhere in the upstream project's history.

Why not a plain diff against the current upstream tip?  Because ORB-SLAM3 froze
these libraries years ago.  Diffing a 2016 fork against a 2024 tip reports every
upstream commit since as if ORB-SLAM3 had made it -- for g2o that is ~15,000
lines of noise that buries the few hundred lines ORB-SLAM3 actually wrote.

So for each vendored file this searches every blob upstream has ever stored
under the same basename, picks the one with the smallest diff, and reports only
the residual.  That residual is, as closely as git can determine it, exactly
what ORB-SLAM3 changed.

  ./tools/upstream_delta.py --project DBoW2
  ./tools/upstream_delta.py --project g2o --out docs/modifications
"""
import argparse
import difflib
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROJECTS = {
    "g2o": dict(
        vendored="reference/ORB_SLAM3/Thirdparty/g2o/g2o",
        upstream="thirdparty/g2o",
    ),
    "Sophus": dict(
        vendored="reference/ORB_SLAM3/Thirdparty/Sophus/sophus",
        upstream="thirdparty/Sophus",
    ),
    "DBoW2": dict(
        vendored="reference/ORB_SLAM3/Thirdparty/DBoW2",
        upstream="thirdparty/DBoW2",
        extra_upstream=["thirdparty/DLib"],   # DUtils/* came from DLib
    ),
}

SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".c"}


def git(repo, *args, binary=False):
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, check=True,
    )
    return out.stdout if binary else out.stdout.decode("utf-8", "replace")


def upstream_blobs_by_basename(repo):
    """basename -> set of blob shas ever stored under that name."""
    index = defaultdict(set)
    # rev-list --objects prints "<sha> [path]" for every reachable object.
    for line in git(repo, "rev-list", "--objects", "--all").splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, path = parts
        base = os.path.basename(path)
        if Path(base).suffix in SUFFIXES:
            index[base].add(sha)
    return index


def normalize(text):
    """Split into lines with CRLF/CR folded to LF.

    ORB-SLAM3 committed several of these files with Windows line endings.  Left
    alone that makes every single line of such a file look modified and buries
    the handful of real changes.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)


def blob_lines(repo, sha):
    try:
        raw = git(repo, "cat-file", "blob", sha, binary=True)
    except subprocess.CalledProcessError:
        return None
    return normalize(raw.decode("utf-8", "replace"))


def closest(repo, shas, target_lines):
    """Blob with the fewest changed lines against target_lines."""
    best = (None, None, 10**9)
    for sha in shas:
        cand = blob_lines(repo, sha)
        if cand is None:
            continue
        sm = difflib.SequenceMatcher(None, cand, target_lines, autojunk=False)
        changed = sum(
            max(i2 - i1, j2 - j1)
            for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag != "equal"
        )
        if changed < best[2]:
            best = (sha, cand, changed)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, choices=sorted(PROJECTS))
    ap.add_argument("--out", default=None,
                    help="directory to write per-file .diff files into")
    args = ap.parse_args()

    cfg = PROJECTS[args.project]
    vendored_dir = ROOT / cfg["vendored"]
    repos = [ROOT / cfg["upstream"]] + [ROOT / p for p in cfg.get("extra_upstream", [])]

    print(f"indexing upstream history: {', '.join(str(r.relative_to(ROOT)) for r in repos)}")
    indexes = [(r, upstream_blobs_by_basename(r)) for r in repos]

    files = sorted(
        p for p in vendored_dir.rglob("*")
        if p.is_file() and p.suffix in SUFFIXES
    )
    print(f"vendored files: {len(files)}\n")

    outdir = Path(args.out) / args.project if args.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    total_delta = 0
    rows = []
    for f in files:
        rel = f.relative_to(vendored_dir)
        target = normalize(f.read_text(encoding="utf-8", errors="replace"))

        best_repo = best_sha = best_lines = None
        best_changed = 10**9
        for repo, index in indexes:
            shas = index.get(f.name)
            if not shas:
                continue
            sha, lines, changed = closest(repo, shas, target)
            if sha and changed < best_changed:
                best_repo, best_sha, best_lines, best_changed = repo, sha, lines, changed

        if best_sha is None:
            rows.append((str(rel), "-", "NO UPSTREAM COUNTERPART", len(target)))
            total_delta += len(target)
            continue

        total_delta += best_changed
        rows.append((str(rel), best_sha[:8],
                     "identical" if best_changed == 0 else f"{best_changed} lines",
                     best_changed))

        if outdir and best_changed:
            diff = difflib.unified_diff(
                best_lines, target,
                fromfile=f"upstream/{rel}  (blob {best_sha[:12]}, closest ancestor)",
                tofile=f"orbslam3/{rel}",
                n=3,
            )
            dest = outdir / (str(rel).replace("/", "__") + ".diff")
            dest.write_text("".join(diff), encoding="utf-8")

    width = max(len(r[0]) for r in rows) + 2
    print(f"{'VENDORED FILE'.ljust(width)}{'CLOSEST BLOB':<14}DELTA")
    print(f"{'-' * (width - 2):<{width}}{'-' * 12:<14}-----")
    for rel, sha, desc, n in sorted(rows, key=lambda r: -r[3]):
        print(f"{rel.ljust(width)}{sha:<14}{desc}")

    changed_files = sum(1 for r in rows if r[3] > 0)
    print()
    print(f"== {args.project}: {changed_files}/{len(rows)} files carry an ORB-SLAM3 delta, "
          f"{total_delta} lines total")
    if outdir:
        print(f"== per-file diffs written to {outdir}")


if __name__ == "__main__":
    sys.exit(main())
