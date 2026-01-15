# sema_diff/denoise.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DenoiseConfig:
    """
    Step-1 (Denoise): whitelist filtering.

    New policy (simplified):
    - Keep only high-signal, structure-changing events:
        - module_added
        - module_removed
        - module_changed
    - Drop everything else (rename, component changes, quality warnings, legacy event types, etc.)

    Compatibility note:
    - We keep the original config fields to avoid breaking imports/usages elsewhere.
    - The old fields are no-ops under this whitelist policy.
    """
    enabled: bool = True

    # 白名单类型
    allowed_types: Tuple[str, ...] = ("module_added", "module_removed", "module_changed")

    # --- legacy fields kept for backward compatibility (no-ops now) ---
    drop_module_renamed: bool = True
    drop_renamed_if_has_other_events_same_pair: bool = True
    keep_rename_if_only: bool = False
    keep_rename_min_jaccard: float = 0.98
    keep_rename_min_module_size: int = 30
    drop_rename_if_unknown_name: bool = True
    unknown_name_keywords: Tuple[str, ...] = ("unknown", "misc", "tmp", "untitled")


def _get_type(ev: Any) -> str:
    """Compatible with both dict events and dataclass-like objects."""
    if isinstance(ev, dict):
        return str(ev.get("type") or "")
    return str(getattr(ev, "type", "") or "")


def _get_id(ev: Any) -> str:
    """Best-effort event id (used for stats)."""
    if isinstance(ev, dict):
        return str(ev.get("id") or "")
    return str(getattr(ev, "id", "") or "")


def denoise_changes(
    changes: List[Any],
    named_a: Any = None,  # kept for signature compatibility; unused
    named_b: Any = None,  # kept for signature compatibility; unused
    cfg: Optional[DenoiseConfig] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Return (filtered_changes, stats)

    Whitelist policy:
    - Only keep events whose type is in cfg.allowed_types.
    - If cfg.enabled is False: return input unchanged.

    This function is intentionally conservative in terms of runtime errors:
    - It never assumes event shape beyond 'type' and optional 'id'.
    - It does not mutate events.
    """
    cfg = cfg or DenoiseConfig()

    if not cfg.enabled:
        return changes, {
            "enabled": False,
            "strategy": "step1_whitelist_disabled",
            "input_events": len(changes),
            "output_events": len(changes),
            "dropped": 0,
            "dropped_ids_sample": [],
            "kept_types": None,
            "config": {"allowed_types": list(getattr(cfg, "allowed_types", ()))},
        }

    allowed = set(getattr(cfg, "allowed_types", ()) or ())
    filtered: List[Any] = []
    dropped_ids: List[str] = []

    for ev in changes:
        t = _get_type(ev)
        if t in allowed:
            filtered.append(ev)
        else:
            eid = _get_id(ev)
            if eid:
                dropped_ids.append(eid)

    stats = {
        "enabled": True,
        "strategy": "step1_type_whitelist",
        "input_events": len(changes),
        "output_events": len(filtered),
        "dropped": len(changes) - len(filtered),
        "dropped_ids_sample": dropped_ids[:20],
        "kept_types": sorted(list(allowed)),
        "config": {
            "allowed_types": sorted(list(allowed)),
        },
    }
    return filtered, stats
