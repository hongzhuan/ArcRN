# sema_diff/denoise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass
class DenoiseConfig:
    """
    Step-1 (Denoise): remove low-value events (especially rename noise).

    Philosophy:
    - Keep structure-changing signals (module_changed / module_added / module_removed / module_component_changed).
    - Rename is mostly low-value in architecture-change reporting. Prefer:
        - Drop rename if the same mapping pair also has module_changed or module_component_changed (rename can be merged later).
        - Otherwise, keep only "high-value renames" (large module + very high jaccard), optional.
    """
    enabled: bool = True

    # Rename filtering
    drop_module_renamed: bool = True
    drop_renamed_if_has_other_events_same_pair: bool = True

    # keep-only heuristics for rename-only pairs
    keep_rename_if_only: bool = False
    keep_rename_min_jaccard: float = 0.98
    keep_rename_min_module_size: int = 30  # max(files_a, files_b) >= this => keep (if enabled)

    # Optional: drop "obviously unhelpful naming churn"
    drop_rename_if_unknown_name: bool = True
    unknown_name_keywords: Tuple[str, ...] = ("unknown", "misc", "tmp", "untitled")


def _base_name(uid: str) -> str:
    # e.g., "Utils#2" -> "Utils"
    return uid.split("#", 1)[0].strip()


def _pair_key(from_uid: str, to_uid: str) -> Tuple[str, str]:
    return (from_uid, to_uid)


def _get_detail(ev: Any) -> Dict[str, Any]:
    # compatible with both dataclass ChangeEvent or dict event
    if isinstance(ev, dict):
        return ev.get("detail", {}) or {}
    return getattr(ev, "detail", {}) or {}


def _get_type(ev: Any) -> str:
    return ev.get("type") if isinstance(ev, dict) else getattr(ev, "type")


def _get_id(ev: Any) -> str:
    return ev.get("id") if isinstance(ev, dict) else getattr(ev, "id")


def _get_conf(ev: Any) -> float:
    return ev.get("confidence") if isinstance(ev, dict) else getattr(ev, "confidence")


def _get_from_to_uids_from_event(ev: Any) -> Optional[Tuple[str, str, Optional[float]]]:
    """
    Try to extract (from_uid, to_uid, jaccard) from:
      - module_renamed.detail: {from_module_uid, to_module_uid, jaccard}
      - module_changed.detail: {from_module_uid, to_module_uid, jaccard}
      - module_component_changed.detail: {from_module_uid, to_module_uid, jaccard}
    """
    d = _get_detail(ev)
    from_uid = d.get("from_module_uid") or d.get("from_uid")
    to_uid = d.get("to_module_uid") or d.get("to_uid")
    if not from_uid or not to_uid:
        return None
    j = d.get("jaccard")
    try:
        j = float(j) if j is not None else None
    except Exception:
        j = None
    return from_uid, to_uid, j


def _module_size(namedclusters_index: Any, module_uid: str) -> int:
    """
    Best-effort get file_count by module_uid from NamedClustersIndex.
    We don't assume a concrete class layout; we try common fields.
    """
    if namedclusters_index is None:
        return 0

    # common: index.module_uid_to_files: Dict[str, Set[str]]
    if hasattr(namedclusters_index, "module_uid_to_files"):
        m = getattr(namedclusters_index, "module_uid_to_files", {}) or {}
        s = m.get(module_uid)
        if s is not None:
            return len(s)

    # common: index.modules list of Module objects (uid, files)
    if hasattr(namedclusters_index, "modules"):
        for m in getattr(namedclusters_index, "modules") or []:
            uid = getattr(m, "uid", None) or getattr(m, "module_uid", None)
            if uid == module_uid:
                files = getattr(m, "files", None) or getattr(m, "file_set", None)
                if files is not None:
                    return len(files)

    # common: index.module_uid_to_module: Dict[str, Module]
    if hasattr(namedclusters_index, "module_uid_to_module"):
        mm = getattr(namedclusters_index, "module_uid_to_module", {}) or {}
        m = mm.get(module_uid)
        if m is not None:
            files = getattr(m, "files", None) or getattr(m, "file_set", None)
            if files is not None:
                return len(files)

    return 0


def _looks_unknown(name: str, keywords: Tuple[str, ...]) -> bool:
    n = (name or "").strip().lower()
    return any(k in n for k in keywords)


def denoise_changes(
    changes: List[Any],
    named_a: Any = None,
    named_b: Any = None,
    cfg: Optional[DenoiseConfig] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Return (filtered_changes, stats)

    Only step-1 implemented: rename noise filtering.
    """
    cfg = cfg or DenoiseConfig()
    if not cfg.enabled or not cfg.drop_module_renamed:
        return changes, {"enabled": cfg.enabled, "dropped": 0, "kept": len(changes)}

    # 1) Build pair -> event types present
    pair_to_types: Dict[Tuple[str, str], Set[str]] = {}
    pair_to_jaccard: Dict[Tuple[str, str], Optional[float]] = {}

    for ev in changes:
        t = _get_type(ev)
        p = _get_from_to_uids_from_event(ev)
        if not p:
            continue
        from_uid, to_uid, j = p
        key = _pair_key(from_uid, to_uid)
        pair_to_types.setdefault(key, set()).add(t)
        if j is not None:
            pair_to_jaccard[key] = j

    filtered: List[Any] = []
    dropped_ids: List[str] = []

    for ev in changes:
        t = _get_type(ev)
        if t != "module_renamed":
            filtered.append(ev)
            continue

        p = _get_from_to_uids_from_event(ev)
        if not p:
            # can't interpret; keep it (conservative)
            filtered.append(ev)
            continue

        from_uid, to_uid, j = p
        key = _pair_key(from_uid, to_uid)

        # Rule A: drop rename if same pair has module_changed or module_component_changed
        if cfg.drop_renamed_if_has_other_events_same_pair:
            other_types = pair_to_types.get(key, set())
            if ("module_changed" in other_types) or ("module_component_changed" in other_types):
                dropped_ids.append(_get_id(ev))
                continue

        # Rule B: drop rename if name looks "unknown"/"misc" (often meaningless churn)
        if cfg.drop_rename_if_unknown_name:
            from_name = _base_name(from_uid)
            to_name = _base_name(to_uid)
            if _looks_unknown(from_name, cfg.unknown_name_keywords) or _looks_unknown(to_name, cfg.unknown_name_keywords):
                dropped_ids.append(_get_id(ev))
                continue

        # Rule C: keep only "high-value rename-only" if enabled; else drop
        if not cfg.keep_rename_if_only:
            dropped_ids.append(_get_id(ev))
            continue

        # If keep_rename_if_only is enabled, apply heuristics
        jj = j if j is not None else pair_to_jaccard.get(key, None)
        if jj is None or jj < cfg.keep_rename_min_jaccard:
            dropped_ids.append(_get_id(ev))
            continue

        size_a = _module_size(named_a, from_uid)
        size_b = _module_size(named_b, to_uid)
        if max(size_a, size_b) < cfg.keep_rename_min_module_size:
            dropped_ids.append(_get_id(ev))
            continue

        filtered.append(ev)

    stats = {
        "enabled": cfg.enabled,
        "strategy": "step1_rename_filter",
        "input_events": len(changes),
        "output_events": len(filtered),
        "dropped": len(dropped_ids),
        "dropped_ids_sample": dropped_ids[:20],
        "config": {
            "drop_module_renamed": cfg.drop_module_renamed,
            "drop_renamed_if_has_other_events_same_pair": cfg.drop_renamed_if_has_other_events_same_pair,
            "keep_rename_if_only": cfg.keep_rename_if_only,
            "keep_rename_min_jaccard": cfg.keep_rename_min_jaccard,
            "keep_rename_min_module_size": cfg.keep_rename_min_module_size,
            "drop_rename_if_unknown_name": cfg.drop_rename_if_unknown_name,
            "unknown_name_keywords": list(cfg.unknown_name_keywords),
        },
    }
    return filtered, stats
