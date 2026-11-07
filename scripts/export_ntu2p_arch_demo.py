"""导出动画架构图页面用的骨架数据：输入 10 帧、GT/FD2/主线输出 50 帧，以及扩散采样过程中状态 z 的解码。

设计：docs/ai/context/20261105-093627-ntu2p-animated-architecture-page-plan.md。
样本与三种输出取自对比页数组（H-val），全部转到场景规范系（+X 为 A→B、+Y 向上），只留身体 22 关节，毫米取整。
采样状态用 BR seed0 在 CPU 上按 ode0（温度 0）重算：z=0 解码即 copy-last，页面据此演示动作如何随 t 出现。
"""

import argparse
import json
from pathlib import Path

import torch

from eval.analyze_ntu2p_naturalness import NTU2P_ACTION_NAMES
from model.forecasting_ntu2p_jitflow import NUM_LOCAL, V_DENOM_MIN, load_ntu2p_jitflow_checkpoint
from utils.ntu2p_canonical import canonical_frame, to_canonical
from utils.ntu2p_naturalness import SMPLX_PARENTS

ARRAYS = "results/forecasting/ntu120_label/ntu2p_bandflow/compare_page/arrays_planner.pt"
BR_CKPT = "save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s0_10000/ema/model000010000.pt"
BODY = 22
STATE_STEPS = (0, 4, 8, 12, 16, 20)


def to_mm(value):
    """[T,2,J,3] 规范系米 -> 嵌套整数列表（毫米），压缩页面体积。"""
    return torch.round(value[:, :, :BODY] * 1000.0).long().tolist()


@torch.no_grad()
def sampling_states(model, obs, action):
    """复刻 NTU2PJiTFlow.sample_ode（Heun、线性 t 网格、自条件），记录每个网格点的状态 z 与每步的 x̂。"""
    prep = model.prepare(obs, action)
    z_root = torch.zeros(1, 2, model.root_k, 3)
    z_local = torch.zeros(1, 2, model.local_k, NUM_LOCAL, 3)
    steps = model.nfe_steps
    grid = torch.linspace(0.0, 1.0, steps + 1)
    states, estimates, sc = [(z_root, z_local)], [], None
    for index in range(steps):
        t, t_next = grid[index], grid[index + 1]
        estimate = model.denoise(prep, z_root, z_local, t.expand(1), sc=sc)
        estimates.append(estimate)
        sc = estimate
        if index == steps - 1:
            z_root, z_local = estimate
            states.append((z_root, z_local))
            break
        denom = (1.0 - t).clamp_min(V_DENOM_MIN)
        v_root, v_local = (estimate[0] - z_root) / denom, (estimate[1] - z_local) / denom
        dt = t_next - t
        euler_root, euler_local = z_root + dt * v_root, z_local + dt * v_local
        second = model.denoise(prep, euler_root, euler_local, t_next.expand(1), sc=sc)
        sc = second
        denom_next = (1.0 - t_next).clamp_min(V_DENOM_MIN)
        z_root = z_root + dt * 0.5 * (v_root + (second[0] - euler_root) / denom_next)
        z_local = z_local + dt * 0.5 * (v_local + (second[1] - euler_local) / denom_next)
        states.append((z_root, z_local))
    decode = lambda pair: model.decode_xyz(prep, pair[0], pair[1])[0][0]  # noqa: E731
    return [decode(state) for state in states], [decode(estimate) for estimate in estimates]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, nargs="+", default=[28, 10])
    parser.add_argument("--output", default="results/forecasting/ntu120_label/ntu2p_bandflow/arch_page/demo.json")
    opts = parser.parse_args()
    torch.set_grad_enabled(False)
    arrays = torch.load(ARRAYS, map_location="cpu")
    model, _ = load_ntu2p_jitflow_checkpoint(BR_CKPT, "cpu")
    model.temperature = 0.0
    cases = []
    for index in opts.cases:
        obs = arrays["obs_xyz"][index : index + 1]
        action = arrays["actions"][index : index + 1]
        frame = canonical_frame(obs, ab_fallback="camera_x")
        canon = lambda value: to_canonical(value.unsqueeze(0), frame)[0]  # noqa: E731
        target = arrays["target_xyz"][index]
        methods = {name: arrays["methods"][key][index] for name, key in (("fd2", "FD2"), ("ode0", "BR_ode0"), ("main", "BR_plan"))}
        states, estimates = sampling_states(model, obs, action)
        # 与对比页导出的 BR:ode0（GPU、分批 16）对照，确认重算的是同一个模型与读出。
        check = (states[-1] - methods["ode0"]).norm(dim=-1).max().item()
        drift = max((estimate - estimates[-1]).norm(dim=-1).max().item() for estimate in estimates)
        mpjpe = {name: (pred - target).norm(dim=-1).mean().item() for name, pred in methods.items()}
        cases.append(
            {
                "index": index,
                "sample_id": arrays["meta"][index]["sample_id"],
                "action_id": int(action[0]),
                "action": NTU2P_ACTION_NAMES[int(action[0])],
                "obs": to_mm(canon(obs[0])),
                "gt": to_mm(canon(target)),
                "fd2": to_mm(canon(methods["fd2"])),
                "main": to_mm(canon(methods["main"])),
                "states": [to_mm(canon(states[k])) for k in STATE_STEPS],
                "state_t": [k / model.nfe_steps for k in STATE_STEPS],
                "mpjpe": {name: round(value, 4) for name, value in mpjpe.items()},
                "check_max_mm": round(check * 1000.0, 3),
                "xhat_max_change_mm": round(drift * 1000.0, 1),
            }
        )
        print("case {} {}：重算 ode0 与导出最大差 {:.3f} mm；20 步内 x̂ 最大变化 {:.1f} mm；mpjpe {}".format(
            index, cases[-1]["action"], check * 1000.0, drift * 1000.0, cases[-1]["mpjpe"]))
    out = Path(opts.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"parents": list(SMPLX_PARENTS[:BODY]), "fps": 20, "cases": cases}
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print("写入 {}（{:.0f} KB）".format(out, out.stat().st_size / 1024))


if __name__ == "__main__":
    main()
