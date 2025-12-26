"""
module_diff_core.py
- 输入：两个版本 NamedClustersIndex + ComponentMapping（可选）+ a2a_jaccard alignment(mapping)
- 输出：模块级 ChangeEvent 列表：
    - module_added / module_removed
    - module_changed（聚合统计，不逐文件）
    - module_renamed（基于 base name 变化）
    - module_component_changed（可选：模块归属 component 变化）

注意：
- 文件变化不再逐条作为事件输出，只作为 module_changed.detail 中的聚合统计 + Top-K 证据列表
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from sema_diff.ir import ChangeEvent, EvidenceItem
from sema_diff.parse_namedclusters import NamedClustersIndex, Module
from sema_diff.parse_clustercomponent import ComponentMapping
from sema_diff.a2a_jaccard import A2AAlignment, ModuleMapping, jaccard as jaccard_fn
from sema_diff.parse_codesem import CodeSemIndex
from sema_diff.parse_archsem import ArchSemIndex


def _base_name(uid: str) -> str:
    # uid = name#occ
    if "#" in uid:
        return uid.split("#", 1)[0]
    return uid


def _index_modules_by_uid(idx: NamedClustersIndex) -> Dict[str, Module]:
    return {m.uid: m for m in idx.modules}


def _top_k(items: List[str], k: int) -> List[str]:
    return items[:k] if len(items) > k else items


def build_module_level_events(
    idx_a: NamedClustersIndex,
    idx_b: NamedClustersIndex,
    comp_a: Optional[ComponentMapping],
    comp_b: Optional[ComponentMapping],
    alignment: A2AAlignment,
    next_id_start: int = 1,
    top_k_files: int = 8,
    min_file_delta: int = 1,
    min_jaccard_to_accept: float = 0.0,
    # NEW: semantic indexes (optional)
    codesem_a: Optional["CodeSemIndex"] = None,
    codesem_b: Optional["CodeSemIndex"] = None,
    archsem_a: Optional["ArchSemIndex"] = None,
    archsem_b: Optional["ArchSemIndex"] = None,
) -> Tuple[List[ChangeEvent], int]:

    """
    min_file_delta:
      - 若某个映射对的 added_files + removed_files < min_file_delta，则不输出 module_changed（用于降噪）
      - 建议默认 1；如果你仍觉得噪声大可以调到 3/5

    min_jaccard_to_accept:
      - 若 mapping.score < 该阈值，则把该映射对当作“不可靠”，可选择不输出 changed/renamed（MVP 默认 0）
    """
    events: List[ChangeEvent] = []
    id_counter = next_id_start

    a_mod = _index_modules_by_uid(idx_a)
    b_mod = _index_modules_by_uid(idx_b)

    # 1) added / removed modules (unmatched)
    for uid in alignment.removed:
        if uid not in a_mod:
            continue
        events.append(
            ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="module_removed",
                confidence=0.95,
                summary=f"Module {uid} removed (unmatched in target version).",
                detail={"module_uid": uid, "module_name": _base_name(uid), "file_count": a_mod[uid].file_count},
                evidence=[EvidenceItem(kind="NamedClusters", ref=f"module:{uid}", note="Present in A, unmatched in B")],
            )
        )
        id_counter += 1

    for uid in alignment.added:
        if uid not in b_mod:
            continue
        events.append(
            ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="module_added",
                confidence=0.95,
                summary=f"Module {uid} added (unmatched from source version).",
                detail={"module_uid": uid, "module_name": _base_name(uid), "file_count": b_mod[uid].file_count},
                evidence=[EvidenceItem(kind="NamedClusters", ref=f"module:{uid}", note="Present in B, unmatched in A")],
            )
        )
        id_counter += 1

    # 2) mapped modules: changed / renamed / component change
    for mm in alignment.mapping:
        if mm.from_uid not in a_mod or mm.to_uid not in b_mod:
            continue

        if mm.score < min_jaccard_to_accept:
            # drop unreliable mapping (optional)
            continue

        ma = a_mod[mm.from_uid]
        mb = b_mod[mm.to_uid]

        files_a = ma.files
        files_b = mb.files

        added_files = sorted(list(files_b - files_a))
        removed_files = sorted(list(files_a - files_b))
        retained_files = sorted(list(files_a & files_b))

        delta = len(added_files) + len(removed_files)

        # 2.1 rename inference (base name different)
        name_a = _base_name(ma.uid)
        name_b = _base_name(mb.uid)
        if name_a != name_b:
            # rename confidence depends on mapping score
            conf = max(0.6, min(0.95, float(mm.score)))
            events.append(
                ChangeEvent(
                    id=f"CHG-{id_counter:04d}",
                    type="module_renamed",
                    confidence=conf,
                    summary=f"Module renamed from {ma.uid} ({name_a}) to {mb.uid} ({name_b}) (mapped by Jaccard).",
                    detail={
                        "from_module_uid": ma.uid,
                        "to_module_uid": mb.uid,
                        "from_name": name_a,
                        "to_name": name_b,
                        "jaccard": round(float(mm.score), 6),
                    },
                    evidence=[
                        EvidenceItem(kind="Derived", ref=f"jaccard={mm.score:.6f}", note="A2A mapping weight"),
                        EvidenceItem(kind="NamedClusters", ref=f"module:{ma.uid}", note="Source module"),
                        EvidenceItem(kind="NamedClusters", ref=f"module:{mb.uid}", note="Target module"),
                    ],
                )
            )
            id_counter += 1

        # 2.2 component change (optional)
        if comp_a is not None and comp_b is not None:
            ca = comp_a.module_uid_to_component.get(ma.uid)
            cb = comp_b.module_uid_to_component.get(mb.uid)
            # 注意：此处是“映射后对比 component”，比旧版“同 uid 对比 component”更合理
            if ca and cb and ca != cb:
                events.append(
                    ChangeEvent(
                        id=f"CHG-{id_counter:04d}",
                        type="module_component_changed",
                        confidence=0.65,
                        summary=f"Module mapped {ma.uid} → {mb.uid} changes component from '{ca}' to '{cb}'.",
                        detail={
                            "from_module_uid": ma.uid,
                            "to_module_uid": mb.uid,
                            "from_component": ca,
                            "to_component": cb,
                            "jaccard": round(float(mm.score), 6),
                        },
                        evidence=[
                            EvidenceItem(kind="ClusterComponent", ref=f"module:{ma.uid}", note="Source component mapping"),
                            EvidenceItem(kind="ClusterComponent", ref=f"module:{mb.uid}", note="Target component mapping"),
                            EvidenceItem(kind="Derived", ref=f"jaccard={mm.score:.6f}", note="A2A mapping weight"),
                        ],
                    )
                )
                id_counter += 1

        # 2.3 module_changed (aggregated)
        if delta >= min_file_delta:
            union_sz = len(files_a | files_b) or 1
            delta_ratio = delta / union_sz
            conf = max(0.55, min(0.95, float(mm.score) * (1.0 - 0.5 * delta_ratio)))

            # === NEW: semantic enrichment (on-demand) ===
            added_top = _top_k(added_files, top_k_files)
            removed_top = _top_k(removed_files, top_k_files)

            code_added = []
            if codesem_b is not None:
                for fp in added_top:
                    desc = codesem_b.file_to_desc.get(fp)
                    if desc:
                        code_added.append({"path": fp, "desc": desc})

            code_removed = []
            if codesem_a is not None:
                for fp in removed_top:
                    desc = codesem_a.file_to_desc.get(fp)
                    if desc:
                        code_removed.append({"path": fp, "desc": desc})

            arch_ctx = {}
            if comp_a is not None and comp_b is not None and (archsem_a is not None or archsem_b is not None):
                ca = comp_a.module_uid_to_component.get(ma.uid)
                cb = comp_b.module_uid_to_component.get(mb.uid)
                if ca:
                    arch_ctx["from_component"] = ca
                    if archsem_a is not None:
                        s = archsem_a.component_to_summary.get(ca)
                        if s:
                            arch_ctx["from_component_summary"] = s
                if cb:
                    arch_ctx["to_component"] = cb
                    if archsem_b is not None:
                        s = archsem_b.component_to_summary.get(cb)
                        if s:
                            arch_ctx["to_component_summary"] = s

                # patterns 作为轻量上下文（可选）
                if archsem_a is not None and archsem_a.patterns:
                    arch_ctx["patterns_a_top"] = archsem_a.patterns[:8]
                if archsem_b is not None and archsem_b.patterns:
                    arch_ctx["patterns_b_top"] = archsem_b.patterns[:8]

            events.append(
                ChangeEvent(
                    id=f"CHG-{id_counter:04d}",
                    type="module_changed",
                    confidence=conf,
                    summary=f"Module {ma.uid} → {mb.uid} changed (added={len(added_files)}, removed={len(removed_files)}, retained={len(retained_files)}; jaccard={mm.score:.3f}).",
                    detail={
                        "from_module_uid": ma.uid,
                        "to_module_uid": mb.uid,
                        "from_name": name_a,
                        "to_name": name_b,
                        "jaccard": round(float(mm.score), 6),
                        "counts": {
                            "added_files": len(added_files),
                            "removed_files": len(removed_files),
                            "retained_files": len(retained_files),
                            "file_count_a": ma.file_count,
                            "file_count_b": mb.file_count,
                            "delta": delta,
                            "delta_ratio": round(delta_ratio, 6),
                        },
                        "examples": {
                            "added_files_top": added_top,
                            "removed_files_top": removed_top,
                        },
                        # NEW: semantics attached to this event
                        "semantics": {
                            "code": {
                                "added_files": code_added,
                                "removed_files": code_removed,
                            },
                            "arch": arch_ctx,
                        },
                    },
                    evidence=[
                        EvidenceItem(kind="Derived", ref=f"jaccard={mm.score:.6f}", note="A2A mapping weight"),
                        EvidenceItem(kind="NamedClusters", ref=f"module:{ma.uid}", note="Source file-set"),
                        EvidenceItem(kind="NamedClusters", ref=f"module:{mb.uid}", note="Target file-set"),
                    ],
                )
            )
            id_counter += 1

    return events, id_counter


def main() -> None:
    """
    右键运行自测（需要你已有 alignment 与两个版本的 JSON 路径）：
    这里只提供结构，建议你通过 run_diff.py 测整体闭环。
    """
    print("module_diff_core.main(): Please test via run_diff.py (module-level pipeline).")


if __name__ == "__main__":
    main()
