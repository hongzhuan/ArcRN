"""
diff_core.py
- MVP 核心 diff：
  1) file universe diff + file_reassigned
  2) module alignment by file-set overlap/Jaccard
  3) infer module_renamed / module_split / module_merge
  4) infer module_moved_between_components
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from sema_diff.config import DiffConfig, default_config
from sema_diff.parse_namedclusters import NamedClustersIndex, Module
from sema_diff.parse_clustercomponent import ComponentMapping

from sema_diff.ir import ChangeEvent, EvidenceItem


@dataclass(frozen=True)
class Snapshot:
    version_label: str
    named: NamedClustersIndex
    comp: ComponentMapping

    # convenience
    files: Set[str]
    modules: List[Module]
    file_to_module: Dict[str, str]
    module_to_component: Dict[str, str]


def build_snapshot(version_label: str, named: NamedClustersIndex, comp: ComponentMapping) -> Snapshot:
    files = set(named.file_to_module_uid.keys())
    module_to_component = comp.module_uid_to_component
    return Snapshot(
        version_label=version_label,
        named=named,
        comp=comp,
        files=files,
        modules=named.modules,
        file_to_module=named.file_to_module_uid,
        module_to_component=module_to_component,
    )


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = min(len(a), len(b))
    return inter / denom if denom > 0 else 0.0


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def diff_file_universe(a: Snapshot, b: Snapshot) -> Tuple[Set[str], Set[str], List[Tuple[str, str, str]]]:
    files_added = b.files - a.files
    files_removed = a.files - b.files

    reassigned: List[Tuple[str, str, str]] = []
    common = a.files & b.files
    for f in sorted(common):
        ma = a.file_to_module.get(f)
        mb = b.file_to_module.get(f)
        if ma and mb and ma != mb:
            reassigned.append((f, ma, mb))
    return files_added, files_removed, reassigned


@dataclass(frozen=True)
class ModulePairScore:
    uid_a: str
    uid_b: str
    overlap: float
    jaccard: float
    inter_count: int


def align_modules(mods_a: List[Module], mods_b: List[Module]) -> List[ModulePairScore]:
    scores: List[ModulePairScore] = []
    for ma in mods_a:
        for mb in mods_b:
            inter = len(ma.files & mb.files)
            ov = _overlap(ma.files, mb.files)
            jac = _jaccard(ma.files, mb.files)
            if inter == 0 and ov == 0.0:
                continue
            scores.append(ModulePairScore(ma.uid, mb.uid, ov, jac, inter))
    # sort high overlap then high jaccard then high inter
    scores.sort(key=lambda s: (s.overlap, s.jaccard, s.inter_count), reverse=True)
    return scores


def greedy_match(scores: List[ModulePairScore]) -> Tuple[List[ModulePairScore], Set[str], Set[str]]:
    matched: List[ModulePairScore] = []
    used_a: Set[str] = set()
    used_b: Set[str] = set()
    for s in scores:
        if s.uid_a in used_a or s.uid_b in used_b:
            continue
        matched.append(s)
        used_a.add(s.uid_a)
        used_b.add(s.uid_b)
    all_a = {s.uid_a for s in scores}
    all_b = {s.uid_b for s in scores}
    unmatched_a = all_a - used_a
    unmatched_b = all_b - used_b
    return matched, unmatched_a, unmatched_b


def _index_modules(mods: List[Module]) -> Dict[str, Module]:
    return {m.uid: m for m in mods}


def infer_module_events(
    a: Snapshot,
    b: Snapshot,
    cfg: DiffConfig,
    next_id_start: int = 1,
) -> Tuple[List[ChangeEvent], int]:
    events: List[ChangeEvent] = []
    id_counter = next_id_start

    scores = align_modules(a.modules, b.modules)
    mod_a = _index_modules(a.modules)
    mod_b = _index_modules(b.modules)

    # 1) rename/equivalence from high-overlap pairs (greedy matching)
    matched, _, _ = greedy_match(scores)

    for p in matched:
        if p.overlap >= cfg.rename_overlap:
            ma = mod_a.get(p.uid_a)
            mb = mod_b.get(p.uid_b)
            if not ma or not mb:
                continue
            if ma.name != mb.name:
                ev = ChangeEvent(
                    id=f"CHG-{id_counter:04d}",
                    type="module_renamed",
                    confidence=min(1.0, max(0.0, p.overlap)),
                    summary=f"Module renamed from {ma.uid} ({ma.name}) to {mb.uid} ({mb.name}) with high file-set overlap.",
                    detail={
                        "from_module_uid": ma.uid,
                        "to_module_uid": mb.uid,
                        "from_name": ma.name,
                        "to_name": mb.name,
                        "overlap": round(p.overlap, 4),
                        "jaccard": round(p.jaccard, 4),
                        "intersect_files": p.inter_count,
                    },
                    evidence=[
                        EvidenceItem(kind="NamedClusters", ref=f"module:{ma.uid}", note="Source module"),
                        EvidenceItem(kind="NamedClusters", ref=f"module:{mb.uid}", note="Target module"),
                        EvidenceItem(kind="Derived", ref=f"overlap={p.overlap:.4f}", note="File-set overlap"),
                    ],
                )
                events.append(ev)
                id_counter += 1

    # 2) split inference: one A overlaps multiple B
    # collect candidate overlaps >= split_merge_overlap
    cand_a_to_bs: Dict[str, List[ModulePairScore]] = defaultdict(list)
    cand_b_to_as: Dict[str, List[ModulePairScore]] = defaultdict(list)
    for s in scores:
        if s.overlap >= cfg.split_merge_overlap:
            cand_a_to_bs[s.uid_a].append(s)
            cand_b_to_as[s.uid_b].append(s)

    # For each A, check split: multiple B with meaningful overlap, and union coverage >= coverage_threshold
    for uid_a, pairs in cand_a_to_bs.items():
        if uid_a not in mod_a:
            continue
        # require at least 2 targets
        if len(pairs) < 2:
            continue
        ma = mod_a[uid_a]
        # choose top K targets (limit to avoid noise)
        pairs_sorted = sorted(pairs, key=lambda x: (x.overlap, x.jaccard, x.inter_count), reverse=True)[:5]
        targets = [mod_b[p.uid_b] for p in pairs_sorted if p.uid_b in mod_b]
        if len(targets) < 2:
            continue

        union_files: Set[str] = set()
        overlaps_list = []
        for p in pairs_sorted:
            if p.uid_b in mod_b:
                union_files |= mod_b[p.uid_b].files
                overlaps_list.append({"to": p.uid_b, "overlap": round(p.overlap, 4), "intersect_files": p.inter_count})
        coverage = (len(ma.files & union_files) / len(ma.files)) if ma.files else 0.0

        if coverage >= cfg.coverage_threshold:
            conf = min(0.85, max(0.3, coverage))  # split is typically less certain than file-level diffs
            ev = ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="module_split",
                confidence=round(conf, 4),
                summary=f"Module {ma.uid} appears split into multiple modules based on file-set overlap/coverage.",
                detail={
                    "from_module_uid": ma.uid,
                    "to_module_uids": [t.uid for t in targets],
                    "coverage": round(coverage, 4),
                    "overlaps": overlaps_list,
                },
                evidence=[
                    EvidenceItem(kind="NamedClusters", ref=f"module:{ma.uid}", note="Source module"),
                    EvidenceItem(kind="Derived", ref=f"coverage={coverage:.4f}", note="Coverage of source files by union of targets"),
                ],
            )
            events.append(ev)
            id_counter += 1

    # 3) merge inference: multiple A overlap one B
    for uid_b, pairs in cand_b_to_as.items():
        if uid_b not in mod_b:
            continue
        if len(pairs) < 2:
            continue
        mb = mod_b[uid_b]
        pairs_sorted = sorted(pairs, key=lambda x: (x.overlap, x.jaccard, x.inter_count), reverse=True)[:5]
        sources = [mod_a[p.uid_a] for p in pairs_sorted if p.uid_a in mod_a]
        if len(sources) < 2:
            continue

        union_files: Set[str] = set()
        overlaps_list = []
        for p in pairs_sorted:
            if p.uid_a in mod_a:
                union_files |= mod_a[p.uid_a].files
                overlaps_list.append({"from": p.uid_a, "overlap": round(p.overlap, 4), "intersect_files": p.inter_count})
        coverage = (len(mb.files & union_files) / len(mb.files)) if mb.files else 0.0

        if coverage >= cfg.coverage_threshold:
            conf = min(0.85, max(0.3, coverage))
            ev = ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="module_merge",
                confidence=round(conf, 4),
                summary=f"Multiple modules appear merged into {mb.uid} based on file-set overlap/coverage.",
                detail={
                    "from_module_uids": [s.uid for s in sources],
                    "to_module_uid": mb.uid,
                    "coverage": round(coverage, 4),
                    "overlaps": overlaps_list,
                },
                evidence=[
                    EvidenceItem(kind="NamedClusters", ref=f"module:{mb.uid}", note="Target module"),
                    EvidenceItem(kind="Derived", ref=f"coverage={coverage:.4f}", note="Coverage of target files by union of sources"),
                ],
            )
            events.append(ev)
            id_counter += 1

    return events, id_counter


def infer_component_moves(a: Snapshot, b: Snapshot, next_id_start: int = 1) -> Tuple[List[ChangeEvent], int]:
    events: List[ChangeEvent] = []
    id_counter = next_id_start

    # consider only module_uids that exist in both snapshots (by uid)
    common_uids = set(m.uid for m in a.modules) & set(m.uid for m in b.modules)
    for uid in sorted(common_uids):
        ca = a.module_to_component.get(uid)
        cb = b.module_to_component.get(uid)
        if not ca or not cb:
            continue
        if ca != cb:
            ev = ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="module_moved_between_components",
                confidence=0.65,  # component mapping is often less stable; tune later
                summary=f"Module {uid} changes associated component from '{ca}' to '{cb}'.",
                detail={"module_uid": uid, "from_component": ca, "to_component": cb},
                evidence=[
                    EvidenceItem(kind="ClusterComponent", ref=f"module:{uid}", note="Occurrence-resolved mapping"),
                ],
            )
            events.append(ev)
            id_counter += 1

    return events, id_counter


def file_events_from_diff(
    files_added: Set[str],
    files_removed: Set[str],
    reassigned: List[Tuple[str, str, str]],
    next_id_start: int = 1,
) -> Tuple[List[ChangeEvent], int]:
    events: List[ChangeEvent] = []
    id_counter = next_id_start

    for f in sorted(files_added):
        events.append(
            ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="file_added",
                confidence=1.0,
                summary=f"Added file {f}.",
                detail={"file": f},
                evidence=[EvidenceItem(kind="NamedClusters", ref=f"file:{f}", note="Present in B, absent in A")],
            )
        )
        id_counter += 1

    for f in sorted(files_removed):
        events.append(
            ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="file_removed",
                confidence=1.0,
                summary=f"Removed file {f}.",
                detail={"file": f},
                evidence=[EvidenceItem(kind="NamedClusters", ref=f"file:{f}", note="Present in A, absent in B")],
            )
        )
        id_counter += 1

    for f, ma, mb in reassigned:
        events.append(
            ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="file_reassigned",
                confidence=0.9,
                summary=f"File {f} reassigned from module {ma} to {mb}.",
                detail={"file": f, "from_module_uid": ma, "to_module_uid": mb},
                evidence=[
                    EvidenceItem(kind="NamedClusters", ref=f"file:{f}", note="Module ownership differs between versions"),
                    EvidenceItem(kind="NamedClusters", ref=f"module:{ma}", note="A module"),
                    EvidenceItem(kind="NamedClusters", ref=f"module:{mb}", note="B module"),
                ],
            )
        )
        id_counter += 1

    return events, id_counter


def main() -> None:
    # 自测：不读真实文件，构造极小结构（建议你主要用 run_diff.py 测整体）
    print("diff_core.main() is intended to be tested via run_diff.py (with real JSON paths).")
    print("OK.")


if __name__ == "__main__":
    main()
