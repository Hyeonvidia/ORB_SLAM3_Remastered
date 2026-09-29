#!/usr/bin/env python3
"""
Draws the README's figure from two summaries of tools/check.sh: ORB-SLAM3 v1.0
and this tree, time per frame and peak memory, run by run. Prints the table
that goes under it.

  ./docker/run.sh -- python3 /workspace/tools/readme_figure.py \
      /workspace/results/check/v1_quick/summary.tsv \
      /workspace/results/check/quick/summary.tsv \
      /workspace/docs/media/at_a_glance.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"kitti_07_mono": "KITTI 07\nmonocular", "kitti_07_stereo": "KITTI 07\nstereo",
         "tum_fr1desk_rgbd": "TUM fr1_desk\nRGB-D", "euroc_V101_stereo": "EuRoC V101\nstereo"}
BEFORE, NOW, INK = "#b9bec6", "#1f6feb", "#30363d"


def read(path):
    rows = {}
    for line in open(path).read().splitlines()[1:]:
        tag, ms, ate, rss = line.split("\t")[:4]
        rows[tag] = tuple(None if v == "-" else float(v) for v in (ms, rss, ate))
    return rows


def table(tags, before, now):
    def cell(a, b, digits, change=True):
        if a is None or b is None:
            return f"– → {b:.{digits}f}" if b is not None else "–"
        return f"{a:.{digits}f} → **{b:.{digits}f}**" + (f" ({100 * (b - a) / a:+.0f} %)" if change else "")
    print("| Run | Tracking per frame, ms | Peak memory, MB | ATE, m |")
    print("|---|---:|---:|---:|")
    for t in tags:
        print(f"| {NAMES[t].replace(chr(10), ' ')} | {cell(before[t][0], now[t][0], 1)} | "
              f"{cell(before[t][1], now[t][1], 0)} | {cell(before[t][2], now[t][2], 3, False)} |")


def panel(ax, title, unit, tags, before, now):
    width = 0.38
    for i, tag in enumerate(tags):
        for value, x, colour in ((before[i], i - width / 2, BEFORE), (now[i], i + width / 2, NOW)):
            if value is None:
                ax.text(x, 0, "not\nreported", ha="center", va="bottom", fontsize=7, color=INK)
                continue
            ax.bar(x, value, width, color=colour)
            ax.text(x, value, f"{value:.1f}" if value < 100 else f"{value:.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
        if before[i] and now[i]:
            ax.text(i, -0.17, f"{100 * (now[i] - before[i]) / before[i]:+.0f} %", ha="center", va="top",
                    fontsize=9, color=NOW, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([NAMES.get(t, t) for t in tags], fontsize=8, color=INK)
    ax.set_title(f"{title}, {unit}", fontsize=10, color=INK, loc="left")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BEFORE)
    ax.tick_params(length=0)
    ax.margins(y=0.15)


def main():
    before, now, out = read(sys.argv[1]), read(sys.argv[2]), sys.argv[3]
    tags = [t for t in NAMES if t in before and t in now]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=160)
    panel(axes[0], "Tracking, per frame", "ms", tags, [before[t][0] for t in tags], [now[t][0] for t in tags])
    panel(axes[1], "Peak memory", "MB", tags, [before[t][1] for t in tags], [now[t][1] for t in tags])
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BEFORE), plt.Rectangle((0, 0), 1, 1, color=NOW)],
               labels=["ORB-SLAM3 v1.0", "this repository"], loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.27, wspace=0.12)
    fig.savefig(out, facecolor="white")
    table(tags, before, now)


if __name__ == "__main__":
    main()
