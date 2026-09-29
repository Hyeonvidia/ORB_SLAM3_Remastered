#!/usr/bin/env python3
"""
Runs a short plan and says, for each run, what a change is judged by: time per
frame, accuracy, what place recognition found, and peak memory. Inside the
container; tools/check.sh writes the plan and calls this.

  check_run.py <out_dir> <jobs> <repeats> [<reference.tsv>]

<out_dir>/plan.txt holds one "tag|binary|arguments" per line, as
tools/dataset_plan.sh prints them. A tag whose sequence is "V101+V102" is a
multi-session run, scored against both ground truths together: they are in one
frame, and the estimate is too once the maps have merged.

Peak memory is the kernel's figure for the process (ru_maxrss), read when it
exits, so the binary needs no option for it.
"""
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_all as E  # noqa: E402  (score() and where the ground truths are)

OUT = Path(sys.argv[1])
JOBS = int(sys.argv[2])
REPEATS = int(sys.argv[3])
REFERENCE = Path(sys.argv[4]) if len(sys.argv) > 4 else None
# Where the binaries are; another build's, to measure it the same way.
BIN = os.environ.get("CHECK_BIN", "/workspace/build/bin")

COUNTED = [("loops", "*Loop detected"), ("merges", "*Merge detected"), ("relocs", "Relocalized!!"),
           ("lost", "Fail to track local map!"), ("maps", "New Map created")]


def run(job):
    repeat, tag, binary, args = job
    d = OUT / f"r{repeat}" / tag
    d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(d / "run.log", "w") as log:
        p = subprocess.Popen([f"{BIN}/{binary}", *args.split()], cwd=d, stdout=log,
                             stderr=subprocess.STDOUT, env=dict(os.environ, ORBSLAM3R_VIEWER="0"))
        _, status, usage = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    row = {"tag": tag, "rc": p.returncode, "secs": time.time() - started, "rss": usage.ru_maxrss / 1024.0}
    text = (d / "run.log").read_text(errors="replace")
    for key in ("median", "mean"):
        found = re.findall(rf"{key} tracking time: ([0-9.eE+-]+)", text)
        row[key] = float(found[-1]) * 1000 if found else None
    for key, phrase in COUNTED:
        row[key] = text.count(phrase)
    result = ate(tag, d)
    row["ate"], row["pairs"] = (result[0], result[1]) if result else (None, 0)
    print(f"  {tag} r{repeat}: rc={row['rc']} {row['secs']:.0f} s", flush=True)
    return row


def ate(tag, d):
    dataset, seq, cfg = re.fullmatch(r"(kitti|euroc|tum)_([^_]+)_(.+)", tag).groups()
    scale = cfg == "mono"  # the only configuration without a metric scale
    if dataset == "kitti":
        gt = E.KITTI_POSES / f"{seq}.txt"
        if cfg == "mono":
            return E.score(d / "KeyFrameTrajectory.txt", gt, scale=True, gt_times=E.KITTI_GRAY / seq / "times.txt")
        return E.score(d / "CameraTrajectory.txt", gt)
    if dataset == "tum":
        name = "KeyFrameTrajectory.txt" if cfg == "mono" else "CameraTrajectory.txt"
        return E.score(d / name, E.TUM_SEQ / "groundtruth.txt", scale=scale)
    if "inertial" in cfg:
        gts = [E.DATASETS / f"EuRoC/{s}/mav0/state_groundtruth_estimate0/data.csv" for s in seq.split("+")]
    else:
        gts = [E.ROOT / f"evaluation/Ground_truth/EuRoC_left_cam/{s}_GT.txt" for s in seq.split("+")]
    gt = gts[0]
    if len(gts) > 1:
        gt = d / "groundtruth.txt"
        gt.write_text("".join(g.read_text() for g in gts))
    return E.score(d / "f_t.txt", gt, scale=scale)


def median(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.median(values) if values else None


def against(value, reference, relative):
    if value is None:
        return "-"
    text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
    if reference:
        text += f" ({100 * (value - reference) / reference:+.1f}%)" if relative else f" ({reference:.3f})"
    return text


def main():
    plan = [line.split("|") for line in (OUT / "plan.txt").read_text().splitlines() if line.strip()]
    jobs = [(r, *p) for r in range(1, REPEATS + 1) for p in plan]
    print(f"== {len(jobs)} runs, {JOBS} at a time -> {OUT}", flush=True)
    with ThreadPoolExecutor(JOBS) as pool:
        rows = list(pool.map(run, jobs))

    reference = {}
    if REFERENCE and REFERENCE.is_file():
        for line in REFERENCE.read_text().splitlines()[1:]:
            tag, ms, err, rss = line.split("\t")[:4]
            reference[tag] = [float(v) if v != "-" else None for v in (ms, err, rss)]

    lines = ["tag\tms\tate\trss\t" + "\t".join(k for k, _ in COUNTED) + "\tpairs\trc"]
    print(f"\n{'RUN':<26}{'ms/frame':>18}{'ATE, m':>18}{'peak RSS, MB':>18}  "
          f"{'loop':>5}{'merge':>6}{'reloc':>6}{'lost':>5}{'maps':>5}{'pairs':>7}  rc")
    for tag in [p[0] for p in plan]:
        mine = [r for r in rows if r["tag"] == tag]
        ms, err, rss = (median(mine, k) for k in ("median", "ate", "rss"))
        ref = reference.get(tag, [None, None, None])
        counts = ["/".join(str(r[k]) for r in mine) for k, _ in COUNTED]
        pairs = "/".join(str(r["pairs"]) for r in mine)
        rc = "/".join(str(r["rc"]) for r in mine)
        print(f"{tag:<26}{against(ms, ref[0], True):>18}{against(err, ref[1], False):>18}"
              f"{against(rss, ref[2], True):>18}  {counts[0]:>5}{counts[1]:>6}{counts[2]:>6}{counts[3]:>5}"
              f"{counts[4]:>5}{pairs:>7}  {rc}")
        lines.append("\t".join(["-" if v is None else f"{v:.4f}" for v in (ms, err, rss)]))
        lines[-1] = f"{tag}\t{lines[-1]}\t" + "\t".join(counts) + f"\t{pairs}\t{rc}"
    (OUT / "summary.tsv").write_text("\n".join(lines) + "\n")
    if reference:
        print(f"\nin brackets: against {REFERENCE.name} -- time and memory as a difference, accuracy as it was")
    return 0 if all(r["rc"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
