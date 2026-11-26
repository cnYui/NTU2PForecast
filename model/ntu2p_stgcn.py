"""FID 用的双人 xyz STGCN 动作识别网络。

结构移植自 ReGenNet `eval/a2m/recognition/models/stgcn.py` 与 `stgcnutils/{tgcn,graph}.py`（源自
ST-GCN，MIT 许可证），层配置、空间划分策略、边重要性加权、两人特征取均值与原网络一致，使 FID 的
特征提取方式与 ReGenNet 相同；只把输入从 rot6d + 平移节点换成关节 xyz，因为本项目各方法只输出 xyz，
没有关节旋转。
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ntu2p_canonical import canonical_frame, to_canonical
from utils.ntu2p_naturalness import SMPLX_PARENTS

# 与 train/train_ntu2p_v2.py 的 LOSS_JOINT_SUBSETS["hand2"] 相同：身体 25 关节 + 每只手 index1/pinky1。
# 各方法都用刚体手，不预测手指；保留 30 个手指关节会让 FID 被各方法同样做不到的手指动作主导。
FID_JOINTS = tuple(range(25)) + (25, 31, 40, 46)
FEATURE_DIM = 256
CANONICAL_AB_FALLBACK = "camera_x"


def subset_parents(joints, parents=SMPLX_PARENTS):
    """子集内每个关节连到它在子集中最近的祖先；返回子集内下标，根为 -1。"""
    position = {joint: index for index, joint in enumerate(joints)}
    result = []
    for joint in joints:
        parent = parents[joint]
        while parent >= 0 and parent not in position:
            parent = parents[parent]
        result.append(position[parent] if parent >= 0 else -1)
    return tuple(result)


def _hop_distance(num_node, edges, max_hop=1):
    adjacency = np.zeros((num_node, num_node))
    for i, j in edges:
        adjacency[j, i] = 1
        adjacency[i, j] = 1
    hop = np.zeros((num_node, num_node)) + np.inf
    reach = np.stack([np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]) > 0
    for d in range(max_hop, -1, -1):
        hop[reach[d]] = d
    return hop


def _normalize_digraph(adjacency):
    degree = np.sum(adjacency, 0)
    inverse = np.diag(np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0))
    return adjacency @ inverse


def spatial_adjacency(parents, center=0, max_hop=1):
    """ST-GCN 'spatial' 划分：自身 / 向心 / 离心三组，与原 Graph(strategy='spatial') 相同。"""
    num_node = len(parents)
    edges = [(i, i) for i in range(num_node)] + [(i, p) for i, p in enumerate(parents) if p >= 0]
    hop = _hop_distance(num_node, edges, max_hop)
    adjacency = np.zeros((num_node, num_node))
    for d in range(max_hop + 1):
        adjacency[hop == d] = 1
    normalized = _normalize_digraph(adjacency)
    groups = []
    for d in range(max_hop + 1):
        root, close, further = (np.zeros((num_node, num_node)) for _ in range(3))
        for i in range(num_node):
            for j in range(num_node):
                if hop[j, i] != d:
                    continue
                if hop[j, center] == hop[i, center]:
                    root[j, i] = normalized[j, i]
                elif hop[j, center] > hop[i, center]:
                    close[j, i] = normalized[j, i]
                else:
                    further[j, i] = normalized[j, i]
        groups.extend([root] if d == 0 else [root + close, further])
    return np.stack(groups)


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x, adjacency):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        return torch.einsum("nkctv,kvw->nctw", (x, adjacency)).contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.0, residual=True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adjacency):
        res = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x, adjacency)) + res)


# 原 STGCN 的 10 层配置（输出通道, 时间步长, 是否残差）。
LAYERS = ((64, 1, False), (64, 1, True), (64, 1, True), (64, 1, True), (128, 2, True),
          (128, 1, True), (128, 1, True), (256, 2, True), (256, 1, True), (256, 1, True))


class XYZSTGCN(nn.Module):
    def __init__(self, num_class=26, joints=FID_JOINTS, coord_dim=3, num_person=2, temporal_kernel=9):
        super().__init__()
        self.joints = tuple(int(j) for j in joints)
        self.num_class = int(num_class)
        self.coord_dim = int(coord_dim)
        self.num_person = int(num_person)
        adjacency = torch.tensor(spatial_adjacency(subset_parents(self.joints)), dtype=torch.float32)
        self.register_buffer("A", adjacency)
        kernel_size = (int(temporal_kernel), int(adjacency.size(0)))
        num_node = len(self.joints)
        # 原网络 num_person=2 时按"两人 × 关节 × 通道"整体做 BatchNorm1d，此处保持。
        self.data_bn = nn.BatchNorm1d(self.num_person * num_node * self.coord_dim)
        blocks, channels = [], self.coord_dim
        for out_channels, stride, residual in LAYERS:
            blocks.append(STGCNBlock(channels, out_channels, kernel_size, stride, residual=residual))
            channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(adjacency.size())) for _ in blocks])
        self.fcn = nn.Conv2d(FEATURE_DIM, self.num_class, kernel_size=1)

    def config(self):
        return OrderedDict([("num_class", self.num_class), ("joints", list(self.joints)), ("coord_dim", self.coord_dim),
                            ("num_person", self.num_person)])

    def forward(self, x):
        """x: [N,T,M,V,C] -> (logits [N,num_class], features [N,256])。"""
        n, t, m, v, c = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(n, m * v * c, t)
        x = self.data_bn(x).view(n, m, v, c, t).permute(0, 1, 3, 4, 2).reshape(n * m, c, t, v)
        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * importance)
        features = F.avg_pool2d(x, x.size()[2:]).view(n, m, -1).mean(dim=1)
        logits = self.fcn(features.view(n, -1, 1, 1)).view(n, -1)
        return logits, features


def recognizer_input(obs_xyz, future_xyz, joints=FID_JOINTS):
    """[B,10,2,55,3] + [B,T,2,55,3] -> 场景规范系下的 [B,10+T,2,V,3]；变换只由观测决定，GT 与各方法共用。"""
    frame = canonical_frame(obs_xyz, ab_fallback=CANONICAL_AB_FALLBACK)
    window = to_canonical(torch.cat((obs_xyz, future_xyz), dim=1), frame)
    return window[:, :, :, list(joints)].contiguous()


def load_recognizer(path, device):
    state = torch.load(path, map_location="cpu")
    config = state["config"]
    model = XYZSTGCN(num_class=config["num_class"], joints=config["joints"], coord_dim=config["coord_dim"],
                     num_person=config["num_person"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()
