"""Skeleton graph topology for COCO 17-keypoint layout.

Standalone re-implementation of pyskl's Graph utility (no mmcv dependency).
Supports 'stgcn_spatial' partitioning used by the STGCN PYSKL checkpoints.
"""

import numpy as np


def _edge2mat(edges, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[j, i] = 1
    return A


def _normalize_digraph(A, dim=0):
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def _hop_distance(num_node, edges, max_hop=1):
    A = np.eye(num_node)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """Skeleton graph for GCN-based action recognition.

    Parameters
    ----------
    layout : str   – 'coco' (17 joints) or 'nturgb+d' (25 joints)
    mode   : str   – 'stgcn_spatial' (distance partitioning) or 'spatial' (identity/in/out)
    max_hop : int  – max adjacency distance (default 1)
    """

    def __init__(self, layout: str = "coco", mode: str = "stgcn_spatial",
                 max_hop: int = 1):
        self.layout = layout
        self.mode = mode
        self.max_hop = max_hop

        self._init_layout(layout)
        self.hop_dis = _hop_distance(self.num_node, self.inward, max_hop)
        self.A = getattr(self, mode)()

    def _init_layout(self, layout: str):
        if layout == "coco":
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12),
                (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6),
                (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2),
            ]
            self.center = 0
        elif layout == "nturgb+d":
            self.num_node = 25
            base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12),
            ]
            self.inward = [(i - 1, j - 1) for i, j in base]
            self.center = 20  # joint 21 (0-indexed)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for i, j in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self) -> np.ndarray:
        """Distance-based partitioning (3 subsets for max_hop=1)."""
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        norm_adj = _normalize_digraph(adj)
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        if self.hop_dis[j, center] >= self.hop_dis[i, center]:
                            a_close[j, i] = norm_adj[j, i]
                        else:
                            a_further[j, i] = norm_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self) -> np.ndarray:
        """Identity / inward / outward partitioning (3 subsets)."""
        Iden = _edge2mat(self.self_link, self.num_node)
        In = _normalize_digraph(_edge2mat(self.inward, self.num_node))
        Out = _normalize_digraph(_edge2mat(self.outward, self.num_node))
        return np.stack((Iden, In, Out))
