#!/usr/bin/env python3
"""
Scores every run under results/ab with eval_all.py's own rules -- Sim(3) for
mono, SE(3) for stereo, the KITTI ground truth matched the way each example
writes its trajectory -- and prints baseline against remaster, per sequence.

  ./docker/run.sh -- python3 /workspace/tools/ab_summarise.py [results/ab]

One column per run, sorted, so the spread is visible next to the median; the
last two columns are the mean tracking time each build reported, in ms.
"""
import re
import statistics
import subprocess
import sys
from pathlib import Path

AB = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/results/ab")
EVAL_ALL = Path("/workspace/tools/eval_all.py")

rows = {}
for d in sorted(AB.glob("*_r[0-9]*")):
    if not d.is_dir():
        continue
    who = d.name.rsplit("_r", 1)[0]
    out = subprocess.run([sys.executable, str(EVAL_ALL), str(d)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"^(\d\d) (mono|stereo)\s+([\d.]+)\s+([\d.]+)m\s+([\d.]+)%\s+(\d+)", line)
        if not m:
            continue
        seq, cfg, rmse = m.group(1), m.group(2), float(m.group(3))
        log = d / f"kitti_{seq}_{cfg}" / "run.log"
        t = None
        if log.is_file():
            mt = re.search(r"mean tracking time: ([\d.eE+-]+)", log.read_text(errors="replace"))
            t = float(mt.group(1)) * 1000 if mt else None
        rows.setdefault((seq, cfg), {}).setdefault(who, []).append((rmse, t))


def col(v):
    return " ".join("%7.2f" % x for x in sorted(v))


def med(v):
    return statistics.median(v) if v else float("nan")


print("%-10s %-26s %-26s %8s %8s %6s %6s" % ("seq", "ATE baseline (m)", "ATE remaster (m)", "med b", "med r", "t_b", "t_r"))
for (seq, cfg), d in sorted(rows.items()):
    b, r = d.get("baseline", []), d.get("remaster", [])
    print("%-10s %-26s %-26s %8.2f %8.2f %6.1f %6.1f" % (
        f"{seq} {cfg}", col([x[0] for x in b]), col([x[0] for x in r]),
        med([x[0] for x in b]), med([x[0] for x in r]),
        med([x[1] for x in b if x[1]]), med([x[1] for x in r if x[1]])))
