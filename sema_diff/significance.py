# sema_diff/significance.py
import math

DEFAULT_WEIGHTS = {
    "struct": 0.45,
    "scope": 0.25,
    "layer": 0.20,
    "semantic": 0.10,
}


def compute_architecture_significance(
    event: dict,
    *,
    max_files_in_project: int,
    weights=DEFAULT_WEIGHTS,
) -> float:
    etype = event["type"]
    detail = event.get("detail", {})

    # ---------- structural impact ----------
    if etype == "module_changed":
        structural = float(detail.get("delta_ratio", 0.0))
    elif etype in ("module_added", "module_removed"):
        structural = 1.0
    elif etype == "module_component_changed":
        structural = 0.6
    else:
        return 0.0

    # ---------- scope impact ----------
    file_count = max(
        detail.get("file_count_a", 0),
        detail.get("file_count_b", 0),
        detail.get("file_count", 0),
    )
    scope = math.log(1 + file_count) / math.log(1 + max_files_in_project)

    # ---------- layer impact ----------
    layer = 0.0
    if etype == "module_component_changed":
        from_c = detail.get("from_component", "")
        to_c = detail.get("to_component", "")
        if from_c != to_c:
            layer = 1.0 if ("Core" in from_c or "Infrastructure" in from_c) else 0.6

    # ---------- semantic impact ----------
    semantic = 0.0
    sem = detail.get("semantics", {})
    code = sem.get("code", {})
    arch = sem.get("arch", {})

    added = len(code.get("added_files", []))
    removed = len(code.get("removed_files", []))
    if added or removed:
        semantic += 0.6 if added >= removed else 0.4

    if arch.get("from_component_summary") != arch.get("to_component_summary"):
        semantic += 0.3

    if arch.get("patterns_a_top") != arch.get("patterns_b_top"):
        semantic += 0.3

    semantic = min(1.0, semantic)

    score = (
        weights["struct"] * structural
        + weights["scope"] * scope
        + weights["layer"] * layer
        + weights["semantic"] * semantic
    )

    return round(float(score), 4)
