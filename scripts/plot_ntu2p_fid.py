"""画 mpjpe–FID 散点图：左图全部方法，右图放大 2/3/4（误差棒为 3 种子标准差）。

配色按 dataviz 参考色板：散点为全部配对比较的图形，只给 3 个重点方法上色（前三槽位在全配对下通过 CVD
校验），其余方法用中性灰并直接标注名称，身份不只靠颜色。
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "copy-last": "copy-last",
    "single": "独立单人 baseline",
    "M1": "1 ReGenNet 移植版（重训）",
    "FD2": "2 FD2",
    "BR": "3 BR:ode0",
    "PG": "4 BR:ode0 + PGait",
}
HIGHLIGHT = {"FD2": "#eb6834", "BR": "#2a78d6", "PG": "#1baf7a"}
NEUTRAL = "#8a8984"
# 右图三点距离近，标签上下错开，避免重叠。
OFFSETS = {"FD2": (8, 8, "left"), "BR": (8, 8, "left"), "PG": (-8, -17, "right")}
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("MPJPE（m，越低越好）", color=TEXT, fontsize=10)
    ax.set_ylabel("FID（越低越好）", color=TEXT, fontsize=10)


def _point(ax, name, row, error_bars, label=True, offset=(7, 5, "left")):
    (x, xs), (y, ys) = row["mpjpe"], row["fid"]
    color = HIGHLIGHT.get(name, NEUTRAL)
    if error_bars and row["num_seeds"] > 1:
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="none", ecolor=color, elinewidth=1.5, capsize=3, zorder=2)
    ax.scatter([x], [y], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
    if label:
        ax.annotate(LABELS.get(name, name), (x, y), xytext=offset[:2], textcoords="offset points", ha=offset[2], fontsize=9, color=TEXT)


def main(args):
    # matplotlib 3.1 的字体缓存里只有 JP 名称；Noto Sans CJK JP 同样覆盖简体中文字形。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with open(args.input) as f:
        rows = json.load(f)["aggregate"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    # 2/3/4 在左图里几乎重合，只在其中心标一次，各自名称放到右图。
    for name, row in rows.items():
        _point(left, name, row, error_bars=False, label=name not in HIGHLIGHT)
    shown = [rows[name] for name in HIGHLIGHT if name in rows]
    if shown:
        center = (sum(r["mpjpe"][0] for r in shown) / len(shown), max(r["fid"][0] for r in shown))
        left.annotate("2 / 3 / 4（见右图）", center, xytext=(7, 8), textcoords="offset points", fontsize=9, color=TEXT)
    for name in HIGHLIGHT:
        if name in rows:
            _point(right, name, rows[name], error_bars=True, offset=OFFSETS.get(name, (7, 5, "left")))
    left.set_ylim(bottom=0)
    for ax, title in ((left, "全部方法"), (right, "放大：2 / 3 / 4（误差棒：3 种子标准差）")):
        _style(ax)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    right.margins(x=0.35, y=0.35)
    fig.suptitle(args.title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.01, "越靠左下越好。FID 由 xyz 版 STGCN 识别网络特征计算，绝对值不能与 ReGenNet 论文比较。",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="MPJPE 与 FID（原协议 test，1253 条）")
    main(parser.parse_args())
