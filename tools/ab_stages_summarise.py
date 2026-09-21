#!/usr/bin/env python3
"""
Prints what tools/ab_stages.sh measured: for every stage of the REGISTER_TIMES
report, the median over the runs of each commit, the difference, and the
difference within each pair of runs.

  python3 tools/ab_stages_summarise.py results/ab_stages/<a>_vs_<b> <a> <b>

Read the pairs, not only the medians: a stage that really changed differs with
the same sign in every pair; one that did not scatters around zero.
"""
import pathlib
import re
import statistics
import sys

STAGES = ["ORB Extraction", "Stereo Matching", "IMU Preintegration", "Pose Prediction", "LM Track",
          "New KF decision", "Total Tracking", "KF Insertion", "MP Culling", "MP Creation", "LBA",
          "KF Culling", "Total Local Mapping"]


def main():
    root, a, b = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    data = {}
    for d in sorted(root.glob("*_r[0-9]*"), key=lambda p: int(p.name.rsplit("_r", 1)[1])):
        who = d.name.rsplit("_r", 1)[0]
        for run in sorted(d.glob("kitti_*")):
            report = run / "ExecMean.txt"
            if not (run / "done").is_file() or not report.is_file():
                continue
            text = report.read_text(errors="replace")
            for stage in STAGES:
                m = re.search(r"^" + re.escape(stage) + r": ([\d.]+)\$", text, re.M)
                if m:
                    data.setdefault(run.name, {}).setdefault(stage, {}).setdefault(who, []).append(float(m.group(1)))
    for tag in sorted(data):
        print("== %s" % tag)
        print("   %-22s %10s %10s %8s   %s" % ("stage, ms", a, b, "diff", "%s - %s, pair by pair" % (b, a)))
        for stage in STAGES:
            v = data[tag].get(stage, {})
            if a not in v or b not in v:
                continue
            ma, mb = statistics.median(v[a]), statistics.median(v[b])
            pairs = " ".join("%+.2f" % (y - x) for x, y in zip(v[a], v[b]))
            print("   %-22s %10.3f %10.3f %+8.3f   %s" % (stage, ma, mb, mb - ma, pairs))
    return 0 if data else 2


if __name__ == "__main__":
    sys.exit(main())
