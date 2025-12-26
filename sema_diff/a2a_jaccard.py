"""
a2a_jaccard.py
- 基于 NamedClusters 的模块->文件集合
- 以 Jaccard 作为边权重做 bipartite maximum weight matching
- 输出：module mapping / unmatched (added/removed) / global summary

依赖策略（自动 fallback）：
1) 优先 igraph（与你现有 a2a.py 一致）
2) 若 igraph 不可用，则尝试 networkx
3) 若都不可用，退化为 greedy（会打印 warning）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path

from sema_diff.config import DiffConfig, default_config
from sema_diff.parse_namedclusters import parse_namedclusters, NamedClustersIndex, Module


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class ModuleMapping:
    from_uid: str
    to_uid: str
    score: float  # jaccard


@dataclass(frozen=True)
class A2AAlignment:
    mapping: List[ModuleMapping]
    removed: List[str]  # in A but unmatched
    added: List[str]    # in B but unmatched
    global_similarity: float
    meta: Dict[str, Any]


def build_module_files(index: NamedClustersIndex) -> Dict[str, Set[str]]:
    """uid -> file set"""
    return {m.uid: set(m.files) for m in index.modules}


def _igraph_max_weight_matching(
    uids_a: List[str],
    uids_b: List[str],
    weights: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int, float]]:
    """
    返回匹配对列表：[(i, j, score)] 其中 i in [0..nA-1], j in [0..nB-1]
    """
    try:
        import igraph  # type: ignore
    except Exception as e:
        raise ImportError(f"igraph not available: {e}")

    nA = len(uids_a)
    nB = len(uids_b)

    g = igraph.Graph()
    g.add_vertices(nA + nB)

    edges = []
    edge_w = []
    for (i, j), w in weights.items():
        # bipartite edge: i in A-side, j in B-side; shift B by nA
        edges.append((i, nA + j))
        edge_w.append(float(w))

    g.add_edges(edges)

    types = [0] * nA + [1] * nB  # bipartite partition
    matching = g.maximum_bipartite_matching(types=types, weights=edge_w)

    pairs: List[Tuple[int, int, float]] = []
    for i in range(nA):
        mate = matching.match_of(i)
        if mate is None:
            continue
        if mate < nA:
            continue
        j = mate - nA
        # find weight for this matched edge (i,j)
        w = weights.get((i, j), 0.0)
        pairs.append((i, j, float(w)))
    return pairs


def _networkx_max_weight_matching(
    uids_a: List[str],
    uids_b: List[str],
    weights: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int, float]]:
    try:
        import networkx as nx  # type: ignore
    except Exception as e:
        raise ImportError(f"networkx not available: {e}")

    nA = len(uids_a)
    nB = len(uids_b)

    G = nx.Graph()
    # add nodes with bipartite attribute
    for i in range(nA):
        G.add_node(("A", i), bipartite=0)
    for j in range(nB):
        G.add_node(("B", j), bipartite=1)

    for (i, j), w in weights.items():
        G.add_edge(("A", i), ("B", j), weight=float(w))

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")

    pairs: List[Tuple[int, int, float]] = []
    for u, v in matching:
        if u[0] == "A":
            i = u[1]
            j = v[1]
        else:
            i = v[1]
            j = u[1]
        w = weights.get((i, j), 0.0)
        pairs.append((i, j, float(w)))
    return pairs


def _greedy_matching(
    uids_a: List[str],
    uids_b: List[str],
    weights: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int, float]]:
    # WARNING: fallback only
    scored = [((i, j), w) for (i, j), w in weights.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    used_a = set()
    used_b = set()
    pairs: List[Tuple[int, int, float]] = []
    for (i, j), w in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, float(w)))
    return pairs


def align_modules_by_jaccard(
    modules_a: Dict[str, Set[str]],
    modules_b: Dict[str, Set[str]],
    min_edge_weight: float = 0.0,
    engine: str = "auto",
) -> A2AAlignment:
    """
    输入：两个版本的 uid->fileset
    输出：A2AAlignment（mapping/added/removed/global_similarity）

    min_edge_weight:
      - 过滤掉 jaccard < min_edge_weight 的边，减少噪声/加速
      - 建议默认 0.0；如果模块很多可以调到 0.05~0.10
    """
    uids_a = list(modules_a.keys())
    uids_b = list(modules_b.keys())
    nA, nB = len(uids_a), len(uids_b)

    # build weights (i,j)->jaccard
    weights: Dict[Tuple[int, int], float] = {}
    for i, ua in enumerate(uids_a):
        fa = modules_a[ua]
        for j, ub in enumerate(uids_b):
            fb = modules_b[ub]
            w = jaccard(fa, fb)
            if w > min_edge_weight:
                weights[(i, j)] = w

    # choose engine
    pairs: List[Tuple[int, int, float]] = []
    used_engine = engine

    if engine == "auto":
        # try igraph -> networkx -> greedy
        try:
            pairs = _igraph_max_weight_matching(uids_a, uids_b, weights)
            used_engine = "igraph"
        except Exception:
            try:
                pairs = _networkx_max_weight_matching(uids_a, uids_b, weights)
                used_engine = "networkx"
            except Exception:
                print("[WARN] igraph/networkx unavailable, falling back to greedy matching (lower quality).")
                pairs = _greedy_matching(uids_a, uids_b, weights)
                used_engine = "greedy"
    elif engine == "igraph":
        pairs = _igraph_max_weight_matching(uids_a, uids_b, weights)
    elif engine == "networkx":
        pairs = _networkx_max_weight_matching(uids_a, uids_b, weights)
    elif engine == "greedy":
        pairs = _greedy_matching(uids_a, uids_b, weights)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    # build mapping
    mapping: List[ModuleMapping] = []
    matched_a = set()
    matched_b = set()
    score_sum = 0.0

    for i, j, w in pairs:
        ua = uids_a[i]
        ub = uids_b[j]
        mapping.append(ModuleMapping(from_uid=ua, to_uid=ub, score=float(w)))
        matched_a.add(ua)
        matched_b.add(ub)
        score_sum += float(w)

    removed = sorted([ua for ua in uids_a if ua not in matched_a])
    added = sorted([ub for ub in uids_b if ub not in matched_b])

    # global similarity (simple, stable): average matched score normalized by max(|A|,|B|)
    denom = max(nA, nB) if max(nA, nB) > 0 else 1
    global_similarity = score_sum / denom

    meta = {
        "engine": used_engine,
        "nA": nA,
        "nB": nB,
        "edges": len(weights),
        "min_edge_weight": min_edge_weight,
    }

    # sort mapping high score first (useful downstream)
    mapping.sort(key=lambda x: x.score, reverse=True)

    return A2AAlignment(
        mapping=mapping,
        removed=removed,
        added=added,
        global_similarity=round(global_similarity, 6),
        meta=meta,
    )


def align_namedclusters_files(
    namedclusters_a_path: Path,
    namedclusters_b_path: Path,
    cfg: Optional[DiffConfig] = None,
    min_edge_weight: float = 0.0,
    engine: str = "auto",
) -> Tuple[A2AAlignment, NamedClustersIndex, NamedClustersIndex]:
    """
    便利函数：直接从 NamedClusters.json 路径读取并对齐
    """
    cfg = cfg or default_config()
    idx_a = parse_namedclusters(namedclusters_a_path, cfg)
    idx_b = parse_namedclusters(namedclusters_b_path, cfg)

    modules_a = build_module_files(idx_a)
    modules_b = build_module_files(idx_b)

    alignment = align_modules_by_jaccard(modules_a, modules_b, min_edge_weight=min_edge_weight, engine=engine)
    return alignment, idx_a, idx_b


def main() -> None:
    """
    右键运行自测：
    - 改下面两个路径为你本机 NamedClusters.json
    - 打印 mapping / added / removed
    """
    a_path = Path(r"C:\path\to\v1.49.0\libuv-v1.49.0_NamedClusters.json")
    b_path = Path(r"C:\path\to\v1.49.1\libuv-v1.49.1_NamedClusters.json")

    cfg = default_config()
    alignment, idx_a, idx_b = align_namedclusters_files(
        a_path, b_path, cfg=cfg, min_edge_weight=0.0, engine="auto"
    )

    print("=== A2A Alignment (Jaccard) ===")
    print("meta:", alignment.meta)
    print("global_similarity:", alignment.global_similarity)
    print("removed(A-only):", len(alignment.removed))
    print("added(B-only):", len(alignment.added))
    print("\nTop-10 mappings:")
    for mm in alignment.mapping[:10]:
        print(f"  {mm.from_uid} -> {mm.to_uid}  score={mm.score:.4f}")


if __name__ == "__main__":
    main()
