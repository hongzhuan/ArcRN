"""
quality.py
- 汇总质量标记与 notes
- 可选：将质量告警也编码为 change events（便于 LLM “必须引用 CHG id”）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from sema_diff.config import DiffConfig, default_config
from sema_diff.diff_core import Snapshot
from sema_diff.ir import ChangeEvent, EvidenceItem


@dataclass(frozen=True)
class QualityReport:
    flags: Dict[str, bool]
    notes: List[str]
    warning_events: List[ChangeEvent]


def build_quality_report(
    a: Snapshot,
    b: Snapshot,
    files_added_count: int,
    files_removed_count: int,
    cfg: DiffConfig,
    next_id_start: int = 1,
) -> Tuple[QualityReport, int]:
    notes: List[str] = []
    flags: Dict[str, bool] = {}

    stable_file_universe = (files_added_count == 0 and files_removed_count == 0 and len(a.files) == len(b.files))
    flags["stable_file_universe"] = stable_file_universe
    if stable_file_universe:
        notes.append(f"File universe unchanged (counts equal: {len(a.files)} → {len(b.files)}).")

    dup_names = bool(a.named.duplicate_module_names or b.named.duplicate_module_names)
    flags["namedcluster_has_duplicate_module_names"] = dup_names
    if dup_names:
        dn = sorted(set(a.named.duplicate_module_names + b.named.duplicate_module_names))
        notes.append(f"Duplicate module names detected: {dn}. Module UIDs use occurrence suffix (e.g., name#1, name#2).")

    empty_mods = bool(a.named.empty_modules or b.named.empty_modules)
    flags["namedcluster_has_empty_module"] = empty_mods
    if empty_mods:
        em = sorted(set(a.named.empty_modules + b.named.empty_modules))
        notes.append(f"Empty modules detected (0 files): {em}.")

    mapping_incomplete = bool(a.comp.unresolved or b.comp.unresolved)
    flags["component_mapping_incomplete"] = mapping_incomplete
    if mapping_incomplete:
        notes.append(
            f"Component mapping unresolved entries exist (A={len(a.comp.unresolved)}, B={len(b.comp.unresolved)}); "
            "component-related diffs may be incomplete."
        )

    # module count delta warning
    ca = len(a.modules)
    cb = len(b.modules)
    delta_ratio = abs(cb - ca) / max(1, ca)
    module_count_delta_large = delta_ratio >= cfg.module_count_delta_warn_ratio
    flags["module_count_delta_large"] = module_count_delta_large
    if module_count_delta_large:
        notes.append(
            f"Module count changes significantly ({ca} → {cb}, delta_ratio={delta_ratio:.2f}); "
            "module partition diffs may reflect clustering instability."
        )

    # Optional: encode warnings as events for traceability
    warning_events: List[ChangeEvent] = []
    id_counter = next_id_start

    def _add_warn(flag_key: str, msg: str) -> None:
        nonlocal id_counter
        warning_events.append(
            ChangeEvent(
                id=f"CHG-{id_counter:04d}",
                type="quality_warning",
                confidence=1.0,
                summary=msg,
                detail={"flag": flag_key},
                evidence=[EvidenceItem(kind="Derived", ref=f"quality:{flag_key}", note="Computed by MVP quality checks")],
            )
        )
        id_counter += 1

    # Only add events for true flags (and meaningful notes)
    if dup_names:
        _add_warn("namedcluster_has_duplicate_module_names", "Duplicate module names detected in NamedClusters outputs.")
    if empty_mods:
        _add_warn("namedcluster_has_empty_module", "Empty modules (0 files) detected in NamedClusters outputs.")
    if mapping_incomplete:
        _add_warn("component_mapping_incomplete", "ClusterComponent mapping contains unresolved entries.")
    if module_count_delta_large:
        _add_warn("module_count_delta_large", "Module count changes significantly between versions; interpret with caution.")
    if stable_file_universe:
        _add_warn("stable_file_universe", "File universe unchanged; file-level diffs are reliable.")

    report = QualityReport(flags=flags, notes=notes, warning_events=warning_events)
    return report, id_counter


def main() -> None:
    print("quality.main() is intended to be tested via run_diff.py (with real JSON paths).")
    print("OK.")


if __name__ == "__main__":
    main()
