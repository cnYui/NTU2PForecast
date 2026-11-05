"""GT / FD2 / 扩散 BR:ode0 三栏并排的对比视频（同一正交机位），供架构对比页使用。

坐标：每个案例用观测求场景规范系（原点 = 观测末帧双人 pelvis 中点，+X = A→B 水平连线，+Y 向上），
三栏共用坐标范围与机位；模型栏叠加淡灰色 GT 作参照。地面上的点是从观测末帧起的踝关节足迹：
成簇 = 迈步，连续拖痕 = 滑行。
输入 .pt 与 eval_ntu2p_v2 --export_arrays / render_ntu2p_review_sheet 相同：obs_xyz、target_xyz、methods、actions、meta。
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch

from eval.analyze_ntu2p_naturalness import NTU2P_ACTION_NAMES
from utils.ntu2p_canonical import canonical_frame, to_canonical

BODY_JOINTS = 22
ANKLES = (7, 8)
FPS = 20
PERSON_COLORS = ("#2F6BFF", "#E5484D")
GT_GHOST = "#9AA0A6"


def load_edges(path="body_models/smplx/SMPLX_NEUTRAL.npz"):
    kintree = np.load(path, allow_pickle=True)["kintree_table"]
    return [(int(kintree[0, child]), child) for child in range(1, BODY_JOINTS) if 0 <= int(kintree[0, child]) < BODY_JOINTS]


def canonical_case(arrays, index, methods):
    obs = arrays["obs_xyz"][index : index + 1].double()
    frame = canonical_frame(obs, ab_fallback="camera_x")
    full = OrderedDict()
    full["GT"] = torch.cat((obs, arrays["target_xyz"][index : index + 1].double()), dim=1)
    for name in methods:
        full[name] = torch.cat((obs, arrays["methods"][name][index : index + 1].double()), dim=1)
    return OrderedDict((name, to_canonical(value, frame)[0, :, :, :BODY_JOINTS].numpy()) for name, value in full.items())


def draw_skeleton(ax, pose, edges, colors, alpha, width):
    for person in range(2):
        for parent, child in edges:
            seg = pose[person, [parent, child]]
            ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=colors[person], alpha=alpha, linewidth=width, solid_capstyle="round")


def render_case(path, poster, motions, labels, edges, obs_len, fps, dpi):
    stack = np.concatenate([m.reshape(-1, 3) for m in motions.values()], axis=0)
    ground = float(stack[:, 1].min()) - 0.03
    center = (stack.max(axis=0) + stack.min(axis=0)) / 2.0
    # matplotlib 3.1 的三维轴按立方体缩放：三个方向取同一边长才不变形；边长由人高与水平活动范围中较大者决定。
    radius = 0.5 * float(max(stack[:, 0].ptp(), stack[:, 2].ptp(), stack[:, 1].max() - ground)) + 0.12
    total = next(iter(motions.values())).shape[0]
    fig = plt.figure(figsize=(4.0 * len(motions), 4.0), dpi=dpi)
    axes = [fig.add_subplot(1, len(motions), k + 1, projection="3d") for k in range(len(motions))]
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=-0.06, wspace=0.0)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    gt = motions["GT"]
    try:
        for frame in range(total):
            future = frame >= obs_len
            for ax, (name, motion) in zip(axes, motions.items()):
                ax.clear()
                ax.set_proj_type("ortho")
                ax.view_init(elev=12, azim=-90)
                ax.dist = 6.6
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[2] - radius, center[2] + radius)
                ax.set_zlim(ground, ground + 2 * radius)
                ax.set_axis_off()
                grid = np.linspace(-radius, radius, 7)
                for g in grid:
                    ax.plot([center[0] + g] * 2, [center[2] - radius, center[2] + radius], [ground] * 2, color=GT_GHOST, alpha=0.25, linewidth=0.6)
                    ax.plot([center[0] - radius, center[0] + radius], [center[2] + g] * 2, [ground] * 2, color=GT_GHOST, alpha=0.25, linewidth=0.6)
                if future:
                    for person in range(2):
                        trail = motion[obs_len - 1 : frame + 1, person][:, list(ANKLES)].reshape(-1, 3)
                        ax.scatter(trail[:, 0], trail[:, 2], np.full(len(trail), ground), s=4, color=PERSON_COLORS[person], alpha=0.5, depthshade=False)
                    if name != "GT":
                        draw_skeleton(ax, gt[frame], edges, (GT_GHOST, GT_GHOST), 0.45, 1.4)
                colors = PERSON_COLORS if future else ("#5B6472", "#5B6472")
                draw_skeleton(ax, motion[frame], edges, colors, 0.95, 2.4)
                ax.set_title(labels[name], fontsize=11, pad=0)
            stamp = "observed  {:.2f} s".format((frame + 1) / FPS) if not future else "predicted  +{:.2f} s".format((frame - obs_len + 1) / FPS)
            fig.suptitle("frame {:2d}/{}   {}".format(frame + 1, total, stamp), fontsize=11, y=0.985)
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(image)
            if frame == obs_len + 25:
                imageio.imwrite(str(poster), image)
    finally:
        writer.close()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cases", required=True, help="JSON：[{index, group}]，group 说明选例来源")
    parser.add_argument("--methods", default="FD2,BR_ode0")
    parser.add_argument("--labels", default="FD2 (DCT regression),BR:ode0 (diffusion)")
    parser.add_argument("--fps", type=int, default=10, help="播放帧率；数据为 20 fps，10 即半速")
    parser.add_argument("--dpi", type=int, default=80)
    opts = parser.parse_args()
    arrays = torch.load(opts.arrays)
    methods = opts.methods.split(",")
    out = Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges = load_edges()
    obs_len = int(arrays["obs_xyz"].shape[1])
    manifest = []
    for case in json.loads(Path(opts.cases).read_text()):
        index = int(case["index"])
        target = arrays["target_xyz"][index].double()
        mpjpe = OrderedDict((name, float((arrays["methods"][name][index].double() - target).norm(dim=-1).mean())) for name in methods)
        labels = OrderedDict([("GT", "GT")])
        for name, label in zip(methods, opts.labels.split(",")):
            labels[name] = "{}  mpjpe {:.3f}".format(label, mpjpe[name])
        meta = arrays["meta"][index]
        stem = "case_{:03d}_{}".format(index, meta["sample_id"])
        render_case(out / (stem + ".mp4"), out / (stem + ".jpg"), canonical_case(arrays, index, methods), labels, edges, obs_len, opts.fps, opts.dpi)
        manifest.append(OrderedDict([("index", index), ("group", case["group"]), ("sample_id", meta["sample_id"]),
                                     ("action", NTU2P_ACTION_NAMES[int(arrays["actions"][index])]), ("mpjpe", mpjpe),
                                     ("video", stem + ".mp4"), ("poster", stem + ".jpg")]))
        print(stem, {k: round(v, 3) for k, v in mpjpe.items()}, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
