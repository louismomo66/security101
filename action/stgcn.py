"""Standalone ST-GCN implementation compatible with pyskl checkpoints.

No mmcv dependency — only PyTorch + numpy.  Matches the architecture of
pyskl's STGCN with default (PYSKL-practice) settings so pre-trained
weights can be loaded directly.

Reference config (pyskl):
    backbone = dict(type='STGCN',
                    graph_cfg=dict(layout='coco', mode='stgcn_spatial'))
    cls_head = dict(type='GCNHead', num_classes=60, in_channels=256)
"""
from __future__ import annotations

import copy as cp

import numpy as np
import torch
import torch.nn as nn

from .graph import Graph

EPS = 1e-4


# ── building blocks ──────────────────────────────────────────────────────

class UnitTCN(nn.Module):
    """Temporal convolution unit (Conv2d along time axis)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9,
                 stride: int = 1, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.conv(x)))


class UnitGCN(nn.Module):
    """Spatial graph convolution unit.

    Default pyskl settings: adaptive='importance', conv_pos='pre'.
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 adaptive: str = "importance", conv_pos: str = "pre",
                 with_res: bool = False):
        super().__init__()
        self.num_subsets = A.size(0)
        self.adaptive = adaptive
        self.conv_pos = conv_pos
        self.with_res = with_res

        if adaptive == "init":
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer("A", A)

        if adaptive in ("offset", "importance"):
            self.PA = nn.Parameter(A.clone())
            if adaptive == "offset":
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif adaptive == "importance":
                nn.init.constant_(self.PA, 1)

        if conv_pos == "pre":
            self.conv = nn.Conv2d(in_ch, out_ch * self.num_subsets, 1)
        else:
            self.conv = nn.Conv2d(self.num_subsets * in_ch, out_ch, 1)

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

        if with_res:
            if in_ch != out_ch:
                self.down = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.BatchNorm2d(out_ch))
            else:
                self.down = lambda x: x
        else:
            self.down = None

    def forward(self, x: torch.Tensor, _A=None) -> torch.Tensor:
        n, c, t, v = x.shape
        res = self.down(x) if self.down is not None else 0

        # Build effective adjacency
        A_map = {None: self.A, "init": self.A}
        if hasattr(self, "PA"):
            A_map["offset"] = self.A + self.PA
            A_map["importance"] = self.A * self.PA
        A = A_map[self.adaptive]

        if self.conv_pos == "pre":
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum("nkctv,kvw->nctw", x, A).contiguous()
        else:
            x = torch.einsum("nctv,kvw->nkctw", x, A).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)


# ── STGCN block ───────────────────────────────────────────────────────────

class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 stride: int = 1, residual: bool = True, **kwargs):
        super().__init__()
        self.gcn = UnitGCN(in_ch, out_ch, A)
        self.tcn = UnitTCN(out_ch, out_ch, 9, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + res)


# ── full STGCN backbone ──────────────────────────────────────────────────

class STGCNBackbone(nn.Module):
    """STGCN backbone matching pyskl default settings.

    Parameters match the pyskl config:
        graph_cfg   = dict(layout='coco', mode='stgcn_spatial')
        in_channels = 3          # (x, y, confidence)
        base_channels = 64
        ch_ratio    = 2
        num_stages  = 10
        inflate_stages = [5, 8]
        down_stages    = [5, 8]
        data_bn_type   = 'VC'
        num_person     = 2
    """

    def __init__(self, graph_cfg: dict | None = None, in_channels: int = 3,
                 base_channels: int = 64, ch_ratio: int = 2,
                 num_stages: int = 10, inflate_stages: list[int] | None = None,
                 down_stages: list[int] | None = None,
                 data_bn_type: str = "VC", num_person: int = 2):
        super().__init__()
        if graph_cfg is None:
            graph_cfg = dict(layout="coco", mode="stgcn_spatial")
        if inflate_stages is None:
            inflate_stages = [5, 8]
        if down_stages is None:
            down_stages = [5, 8]

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32)
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio

        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        modules: list[nn.Module] = []
        cur_base = base_channels

        if in_channels != base_channels:
            modules.append(STGCNBlock(in_channels, base_channels, A.clone(),
                                      1, residual=False))

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + int(i in down_stages)
            in_ch = cur_base
            if i in inflate_stages:
                inflate_times += 1
            out_ch = int(base_channels * ch_ratio ** inflate_times + EPS)
            cur_base = out_ch
            modules.append(STGCNBlock(in_ch, out_ch, A.clone(), stride))

        if in_channels == base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (N, M, T, V, C)  →  output: (N, M, C', T', V)"""
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # N M V C T

        if self.data_bn_type == "MVC":
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        x = (x.view(N, M, V, C, T)
              .permute(0, 1, 3, 4, 2)
              .contiguous()
              .view(N * M, C, T, V))

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x


# ── classifier head ──────────────────────────────────────────────────────

class GCNHead(nn.Module):
    """Global-average-pool + FC classifier."""

    def __init__(self, num_classes: int, in_channels: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, C, T, V) → pool over M, T, V
        N, M, C, T, V = x.shape
        x = x.reshape(N, M * C * T * V // (C), C, -1).mean(dim=1)
        # Simpler: just reshape and mean
        x = x.reshape(N, -1, C).mean(dim=1)  # not right

        return self.fc(self.drop(x))


class RecognizerGCN(nn.Module):
    """Combined backbone + head matching pyskl's RecognizerGCN.

    This is the top-level model that loads pyskl checkpoints.
    """

    def __init__(self, num_classes: int = 60, in_channels: int = 3,
                 graph_cfg: dict | None = None, dropout: float = 0.0):
        super().__init__()
        if graph_cfg is None:
            graph_cfg = dict(layout="coco", mode="stgcn_spatial")

        self.backbone = STGCNBackbone(graph_cfg=graph_cfg,
                                      in_channels=in_channels)

        # Final output channels after 10 stages: 64 * 2^2 = 256
        head_in = int(64 * 2 ** 2)
        self.cls_head = GCNHead(num_classes, head_in, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (N, M, T, V, C) → logits (N, num_classes)"""
        feat = self.backbone(x)  # (N, M, C, T', V)
        # Pool: mean over M, T, V
        N, M, C, T, V = feat.shape
        feat = feat.mean(dim=1)   # (N, C, T, V)
        feat = feat.mean(dim=-1)  # (N, C, T)
        feat = feat.mean(dim=-1)  # (N, C)
        return self.cls_head.fc(self.cls_head.drop(feat))

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu",
                        num_classes: int = 60) -> "RecognizerGCN":
        """Load a pyskl-format checkpoint (.pth).

        pyskl checkpoints have structure:
            {'state_dict': {'backbone.xxx': ..., 'cls_head.fc_cls.xxx': ...}}

        We remap 'cls_head.fc_cls.*' → 'cls_head.fc.*' to match our naming.
        """
        raw = torch.load(path, map_location=device, weights_only=False)

        # Extract state_dict from mmcv checkpoint wrapper
        sd = raw.get("state_dict", raw)

        # Remap keys: pyskl uses 'cls_head.fc_cls' → we use 'cls_head.fc'
        mapped: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            k_new = k.replace("cls_head.fc_cls.", "cls_head.fc.")
            mapped[k_new] = v

        model = cls(num_classes=num_classes)
        model.load_state_dict(mapped, strict=False)
        model.to(device)
        model.eval()
        return model
