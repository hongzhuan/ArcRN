"""
parse_clustercomponent.py
- 解析 SemArc 输出的 *_ClusterComponent.json
- 输入：ClusterComponent.json + NamedClustersIndex.name_to_uids_queue
- 输出：
  1) module_uid_to_component: Dict[module_uid, component_name]
  2) component_to_module_uids: Dict[component_name, List[module_uid]]
  3) unresolved: List[UnresolvedClusterRef]  # 无法消歧/找不到对应 module_uid 的情况
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, deque

from sema_diff.config import DiffConfig, default_config
from sema_diff.parse_namedclusters import NamedClustersIndex, parse_namedclusters


@dataclass(frozen=True)
class UnresolvedClusterRef:
    component: str
    cluster_name: str
    reason: str


@dataclass(frozen=True)
class ComponentMapping:
    module_uid_to_component: Dict[str, str]
    component_to_module_uids: Dict[str, List[str]]
    unresolved: List[UnresolvedClusterRef]

    raw_name: Optional[str] = None
    schema_version: Optional[str] = None


def _copy_queues(name_to_uids_queue: Dict[str, "deque[str]"]) -> Dict[str, "deque[str]"]:
    copied: Dict[str, "deque[str]"] = {}
    for k, q in name_to_uids_queue.items():
        copied[k] = deque(q)  # shallow copy is sufficient (elements are strings)
    return copied


def parse_clustercomponent(
    json_path: Path,
    name_to_uids_queue: Dict[str, "deque[str]"],
    cfg: DiffConfig,
) -> ComponentMapping:
    if not json_path.exists():
        raise FileNotFoundError(f"ClusterComponent json not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    schema_version = data.get("@schemaVersion")
    raw_name = data.get("name")

    structure = data.get("structure")
    if not isinstance(structure, list):
        raise ValueError("Invalid ClusterComponent.json: top-level 'structure' must be a list.")

    # Work on a copy to avoid consuming caller's queues
    queues = _copy_queues(name_to_uids_queue) if cfg.enable_occurrence_disambiguation else {}

    module_uid_to_component: Dict[str, str] = {}
    component_to_module_uids: Dict[str, List[str]] = defaultdict(list)
    unresolved: List[UnresolvedClusterRef] = []

    for comp_node in structure:
        if not isinstance(comp_node, dict):
            continue
        if comp_node.get("@type") != "component":
            continue

        comp_name = str(comp_node.get("name", "")).strip()
        nested = comp_node.get("nested", [])

        if not isinstance(nested, list):
            continue

        for cl in nested:
            if not isinstance(cl, dict):
                continue
            if cl.get("@type") != "cluster":
                continue

            cluster_name = cl.get("name")
            if not isinstance(cluster_name, str):
                continue
            cluster_name = cluster_name.strip()

            # resolve cluster_name -> module_uid
            module_uid: Optional[str] = None
            if cfg.enable_occurrence_disambiguation:
                q = queues.get(cluster_name)
                if q and len(q) > 0:
                    module_uid = q.popleft()
                else:
                    unresolved.append(
                        UnresolvedClusterRef(
                            component=comp_name,
                            cluster_name=cluster_name,
                            reason="No remaining occurrence in NamedClusters queue for this cluster name.",
                        )
                    )
                    continue
            else:
                # fallback: no disambiguation, use raw name as uid (not recommended)
                module_uid = cluster_name

            # record mapping
            # if duplicated mapping appears, keep the first and mark unresolved as warning
            if module_uid in module_uid_to_component and module_uid_to_component[module_uid] != comp_name:
                unresolved.append(
                    UnresolvedClusterRef(
                        component=comp_name,
                        cluster_name=cluster_name,
                        reason=f"Module UID {module_uid} already mapped to {module_uid_to_component[module_uid]}.",
                    )
                )
                continue

            module_uid_to_component[module_uid] = comp_name
            component_to_module_uids[comp_name].append(module_uid)

    # freeze defaultdict to normal dict
    component_to_module_uids_final = {k: v for k, v in component_to_module_uids.items()}

    return ComponentMapping(
        module_uid_to_component=module_uid_to_component,
        component_to_module_uids=component_to_module_uids_final,
        unresolved=unresolved,
        raw_name=raw_name,
        schema_version=schema_version,
    )


def main() -> None:
    """
    右键运行本文件进行自测：
    - 填两个 json 路径：NamedClusters + ClusterComponent（同一版本）
    - 程序会先解析 NamedClusters，再解析 ClusterComponent 并进行 occurrence 消歧。
    """
    cfg = default_config()

    # TODO: 改成你的实际路径（同一个版本）
    namedclusters_path = Path(r"..\sema_results\libuv-1.49.0")
    clustercomponent_path = Path(r"..\sema_results\libuv-1.49.0")

    # 支持只给目录
    if namedclusters_path.is_dir():
        candidates = list(namedclusters_path.rglob("*_NamedClusters.json"))
        if not candidates:
            raise RuntimeError(f"No *_NamedClusters.json found under {namedclusters_path}")
        candidates.sort(key=lambda p: (len(str(p)), str(p)))
        namedclusters_path = candidates[0]

    if clustercomponent_path.is_dir():
        candidates = list(clustercomponent_path.rglob("*_ClusterComponent.json"))
        if not candidates:
            raise RuntimeError(f"No *_ClusterComponent.json found under {clustercomponent_path}")
        candidates.sort(key=lambda p: (len(str(p)), str(p)))
        clustercomponent_path = candidates[0]

    idx: NamedClustersIndex = parse_namedclusters(namedclusters_path, cfg)
    mapping = parse_clustercomponent(clustercomponent_path, idx.name_to_uids_queue, cfg)

    print("=== parse_clustercomponent.main() ===")
    print(f"NamedClusters: {namedclusters_path}")
    print(f"ClusterComponent: {clustercomponent_path}")
    print(f"Components mapped: {len(mapping.component_to_module_uids)}")
    print(f"Module->Component mapped: {len(mapping.module_uid_to_component)}")
    print(f"Unresolved refs: {len(mapping.unresolved)}")

    # show first 5 component mappings
    print("\nFirst 5 components:")
    items = list(mapping.component_to_module_uids.items())
    for comp, mods in items[:5]:
        print(f"  {comp}: {mods[:10]}{' ...' if len(mods) > 10 else ''}")

    if mapping.unresolved:
        print("\nFirst 10 unresolved refs:")
        for u in mapping.unresolved[:10]:
            print(f"  component={u.component} cluster={u.cluster_name} reason={u.reason}")


if __name__ == "__main__":
    main()
