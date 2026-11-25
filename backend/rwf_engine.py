"""
Exact Random Walk Fingerprint (RWF) Algorithm 1 implementation in NumPy + SciPy.
Reference: Li, Wang, & Böhm, "Scalable Graph Classification via Random Walk Fingerprints", IEEE ICDM 2024.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import sparse


FEATURE_VOCAB = ["C", "N", "O"]


def _build_adjacency(
    nodes: List[Dict[str, Any]], edges: List[List[str]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], np.ndarray, np.ndarray]:
    """Return ordered nodes, degree map, dense adjacency A, and degree array."""
    deg = {nd["id"]: 0 for nd in nodes}
    valid_edges: List[Tuple[str, str]] = []
    seen = set()
    for e in edges:
        if len(e) < 2:
            continue
        u, v = str(e[0]), str(e[1])
        if u == v or u not in deg or v not in deg:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        valid_edges.append((u, v))
        deg[u] += 1
        deg[v] += 1

    return nodes, deg, valid_edges


def partition_nodes(
    nodes: List[Dict[str, Any]],
    edges: List[List[str]],
    mode: str = "rwf-d",
    k: int = 2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[List[str]], Dict[str, int], Dict[str, int]]:
    """
    Partition graph nodes by structural role:
    - rwf-d: Degree heuristic (sort descending, split into k equal chunks)
    - rwf-cp: Core-periphery heuristic (KM-config modularity-style quality optimization, k=2)
    - rwf-w: Whole graph (no split, k=1)
    - rwf-r: Random split (ablation sanity check)
    """
    _, deg, valid_edges = _build_adjacency(nodes, edges)
    n = len(nodes)
    if n == 0:
        return [], [], {}, {}

    if mode == "rwf-w":
        ordered = sorted(nodes, key=lambda nd: (-deg[nd["id"]], nd["id"]))
        groups = [[nd["id"] for nd in ordered]]
        node_group = {nd["id"]: 0 for nd in ordered}
        return ordered, groups, node_group, deg

    if mode == "rwf-r":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        ordered = [nodes[i] for i in perm]
        k_eff = max(1, min(k, n))
        groups: List[List[str]] = [[] for _ in range(k_eff)]
        node_group: Dict[str, int] = {}
        for idx, nd in enumerate(ordered):
            g = min(k_eff - 1, (idx * k_eff) // n)
            groups[g].append(nd["id"])
            node_group[nd["id"]] = g
        return ordered, groups, node_group, deg

    if mode == "rwf-cp":
        # Kojaku & Masuda (KM-config) inspired core-periphery quality maximization:
        # Q_cp = sum_{i,j} (A_ij - (d_i * d_j) / (2m)) * (x_i + x_j - x_i * x_j)
        ordered = sorted(nodes, key=lambda nd: (-deg[nd["id"]], nd["id"]))
        id_to_idx = {nd["id"]: i for i, nd in enumerate(ordered)}
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in valid_edges:
            iu, iv = id_to_idx[u], id_to_idx[v]
            A[iu, iv] = 1.0
            A[iv, iu] = 1.0
        d_vec = A.sum(axis=1)
        two_m = max(1.0, float(d_vec.sum()))
        null_mat = np.outer(d_vec, d_vec) / two_m
        B = A - null_mat

        # Greedy coordinate ascent (t = 10 passes, matching §3.10 O(t*m))
        x = (d_vec >= np.mean(d_vec)).astype(np.int32)
        if x.sum() == 0:
            x[0] = 1
        if x.sum() == n and n > 1:
            x[-1] = 0

        def cp_score(vec: np.ndarray) -> float:
            mask = np.clip(vec[:, None] + vec[None, :], 0, 1)
            return float(np.sum(B * mask))

        best_q = cp_score(x)
        for _ in range(10):
            improved = False
            for i in range(n):
                if (x[i] == 1 and x.sum() <= 1) or (x[i] == 0 and x.sum() >= n - 1):
                    continue
                x[i] = 1 - x[i]
                q = cp_score(x)
                if q > best_q + 1e-9:
                    best_q = q
                    improved = True
                else:
                    x[i] = 1 - x[i]
            if not improved:
                break

        core = [ordered[i]["id"] for i in range(n) if x[i] == 1]
        periph = [ordered[i]["id"] for i in range(n) if x[i] == 0]
        ordered = [nd for nd in ordered if nd["id"] in core] + [
            nd for nd in ordered if nd["id"] in periph
        ]
        groups = [core, periph] if periph else [core]
        node_group = {}
        for gi, grp in enumerate(groups):
            for nid in grp:
                node_group[nid] = gi
        return ordered, groups, node_group, deg

    # Default: RWF-D (Degree heuristic)
    ordered = sorted(nodes, key=lambda nd: (-deg[nd["id"]], nd["id"]))
    k_eff = max(1, min(k, n))
    groups = [[] for _ in range(k_eff)]
    node_group = {}
    for idx, nd in enumerate(ordered):
        g = min(k_eff - 1, (idx * k_eff) // n)
        groups[g].append(nd["id"])
        node_group[nd["id"]] = g
    return ordered, groups, node_group, deg


def compute_rwf(
    nodes: List[Dict[str, Any]],
    edges: List[List[str]],
    mode: str = "rwf-d",
    k: int = 2,
    tau_c: int = 3,
    use_node_features: bool = False,
    include_matrices: bool = True,
) -> Dict[str, Any]:
    """
    Execute Algorithm 1 (sparse matrix powers F <- T @ F) and compute:
    - Eq. 1: loop^tau_{V_i}
    - Eq. 2: walk^tau_{V_i}
    - Eq. 3: walk^tau_{V_i -> V_j}
    - Eq. 8 & 9 (optional): P, Q node feature aggregations (xl, xw)
    - Spectral eigenvalues of T (Eq. 4-5) for theoretical verification
    """
    t0 = time.perf_counter()
    ordered, groups, node_group, deg = partition_nodes(nodes, edges, mode=mode, k=k)
    n = len(ordered)
    id_to_idx = {nd["id"]: i for i, nd in enumerate(ordered)}

    # Build sparse symmetric normalized transition matrix T = D^(-1/2) A D^(-1/2)
    rows, cols, vals = [], [], []
    seen = set()
    for e in edges:
        if len(e) < 2:
            continue
        u, v = str(e[0]), str(e[1])
        if u == v or u not in id_to_idx or v not in id_to_idx:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        iu, iv = id_to_idx[u], id_to_idx[v]
        if deg[u] > 0 and deg[v] > 0:
            w = 1.0 / np.sqrt(float(deg[u] * deg[v]))
            rows.extend([iu, iv])
            cols.extend([iv, iu])
            vals.extend([w, w])

    T_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    T_dense = T_sparse.toarray()

    # Exact eigenvalues of T (Eq. 4) for small/medium graphs
    eigenvalues = (
        np.sort(np.linalg.eigvalsh(T_dense))[::-1].round(4).tolist()
        if 0 < n <= 200
        else []
    )

    # One-hot node feature matrix X (n x p)
    p = len(FEATURE_VOCAB)
    X = np.zeros((n, p), dtype=np.float64)
    for i, nd in enumerate(ordered):
        feat = nd.get("feat", "C")
        if feat in FEATURE_VOCAB:
            X[i, FEATURE_VOCAB.index(feat)] = 1.0
        else:
            X[i, 0] = 1.0

    group_indices = [[id_to_idx[nid] for nid in grp] for grp in groups]
    k_eff = len(groups)

    F = np.eye(n, dtype=np.float64)
    powers: List[List[List[float]]] = []
    steps: List[Dict[str, Any]] = []
    flat_vector: List[float] = []

    for tau in range(1, max(1, tau_c) + 1):
        # Line 11 of Algorithm 1: F <- T @ F (sparse x dense)
        F = T_sparse.dot(F)
        if include_matrices:
            powers.append(np.round(F, 4).tolist())

        step_items: List[Dict[str, Any]] = []

        # Eq. 1 (loop) & Eq. 2 (walk within)
        for gi in range(k_eff):
            idxs = group_indices[gi]
            sz = len(idxs)
            if sz > 0:
                sub = F[np.ix_(idxs, idxs)]
                loop_val = float(np.trace(sub) / sz)
                walk_val = float(np.sum(sub) / (sz * sz))
            else:
                loop_val, walk_val = 0.0, 0.0

            role_type = "hub" if gi == 0 else ("periph" if gi == 1 else "mid")
            step_items.append({"label": f"loop V{gi + 1}", "val": round(loop_val, 6), "type": role_type})
            step_items.append({"label": f"walk V{gi + 1}", "val": round(walk_val, 6), "type": role_type})
            flat_vector.extend([loop_val, walk_val])

        # Eq. 3 (walk between pairs of groups)
        for gi in range(k_eff):
            for gj in range(gi + 1, k_eff):
                idxs_i = group_indices[gi]
                idxs_j = group_indices[gj]
                if idxs_i and idxs_j:
                    sub_bw = F[np.ix_(idxs_i, idxs_j)]
                    bw_val = float(np.sum(sub_bw) / (len(idxs_i) * len(idxs_j)))
                else:
                    bw_val = 0.0
                step_items.append({"label": f"V{gi + 1}→V{gj + 1}", "val": round(bw_val, 6), "type": "between"})
                flat_vector.append(bw_val)

        # Eq. 8 & 9 (optional node features: P and Q)
        if use_node_features:
            for gi in range(k_eff):
                idxs = group_indices[gi]
                sz = len(idxs)
                if sz > 0:
                    X_g = X[idxs, :].T  # p x |V_i|
                    sub = F[np.ix_(idxs, idxs)]
                    P_mat = (1.0 / sz) * (X_g @ np.diag(np.diag(sub)) @ X_g.T)
                    Q_mat = (1.0 / (sz * sz)) * (X_g @ sub @ X_g.T)
                    xl_vec = P_mat.mean(axis=1)
                    xw_vec = Q_mat.mean(axis=1)
                else:
                    xl_vec = np.zeros(p)
                    xw_vec = np.zeros(p)

                for fi, fname in enumerate(FEATURE_VOCAB):
                    step_items.append({"label": f"xl({fname}) V{gi + 1}", "val": round(float(xl_vec[fi]), 6), "type": "feat"})
                    step_items.append({"label": f"xw({fname}) V{gi + 1}", "val": round(float(xw_vec[fi]), 6), "type": "feat"})
                    flat_vector.extend([float(xl_vec[fi]), float(xw_vec[fi])])

        steps.append({"tau": tau, "items": step_items})

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ordered": ordered,
        "degrees": deg,
        "groups": groups,
        "node_group": node_group,
        "powers": powers,
        "steps": steps,
        "fingerprint": [round(v, 6) for v in flat_vector],
        "dimension": len(flat_vector),
        "eigenvalues": eigenvalues,
        "elapsed_ms": round(elapsed_ms, 3),
        "meta": {
            "n": n,
            "m": len(seen),
            "k": k_eff,
            "tau_c": tau_c,
            "mode": mode,
            "use_node_features": use_node_features,
        },
    }


def compare_fingerprints(vec_a: List[float], vec_b: List[float]) -> Dict[str, Any]:
    """Compute exact L2 distance, L1 distance, and cosine similarity between two fingerprints."""
    L = min(len(vec_a), len(vec_b))
    if L == 0:
        return {"l2_distance": 0.0, "l1_distance": 0.0, "cosine_similarity": 1.0, "deltas": []}
    a = np.array(vec_a[:L], dtype=np.float64)
    b = np.array(vec_b[:L], dtype=np.float64)
    diff = a - b
    l2 = float(np.linalg.norm(diff))
    l1 = float(np.sum(np.abs(diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
    return {
        "l2_distance": round(l2, 4),
        "l1_distance": round(l1, 4),
        "cosine_similarity": round(cos, 4),
        "deltas": [round(float(x), 4) for x in diff],
    }
