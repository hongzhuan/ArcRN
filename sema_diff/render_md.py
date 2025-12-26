"""
render_md.py
- 从 DiffIR（或 diff_ir.json）生成 Markdown
- 支持两种模式：
  1) template：不使用 LLM，稳定输出
  2) llm：调用 deepseek-chat，生成更自然的报告（强约束不幻觉）

注意：LLM 输入建议精简，避免 token 浪费与上下文限制。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from sema_diff.llm import generate_markdown_from_ir


FILE_TYPES = {"file_added", "file_removed", "file_reassigned"}  # 现在一般会为空
MODULE_TYPES = {
    "module_added",
    "module_removed",
    "module_changed",
    "module_renamed",
    "module_split",
    "module_merge",
    "module_component_changed",
}
COMP_TYPES = {"module_moved_between_components"}  # 老事件，当前模块级管线一般不再生成
QUALITY_TYPES = {"quality_warning"}


def load_ir_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truncate_text(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def slim_ir_for_llm(
    ir: Dict[str, Any],
    max_evidence_note_len: int = 160,
    max_desc_len: int = 260,
    max_arch_summary_len: int = 320,
) -> Dict[str, Any]:
    """
    精简 IR：保留 meta/quality/entities/changes 的核心字段，截断 evidence note 与语义长文本。
    目标：
    - 保留 detail（尤其是 detail.semantics），让 LLM 能写“语义变化”
    - 但限制 CodeSem/ArchSem 摘要长度，避免 token 爆炸
    """
    meta = ir.get("meta", {})
    quality = ir.get("quality", {})
    entities = ir.get("entities", {})

    changes_in = ir.get("changes", [])
    changes_out = []
    for ev in changes_in:
        if not isinstance(ev, dict):
            continue

        detail = ev.get("detail", {}) or {}

        # --- truncate semantics fields if present ---
        sem = (detail.get("semantics") or {})
        if isinstance(sem, dict):
            code = sem.get("code") or {}
            if isinstance(code, dict):
                # truncate added_files/removed_files desc
                for key in ("added_files", "removed_files"):
                    arr = code.get(key)
                    if isinstance(arr, list):
                        for item in arr:
                            if isinstance(item, dict) and isinstance(item.get("desc"), str):
                                item["desc"] = _truncate_text(item["desc"], max_desc_len)

            arch = sem.get("arch") or {}
            if isinstance(arch, dict):
                for k in ("from_component_summary", "to_component_summary"):
                    if isinstance(arch.get(k), str):
                        arch[k] = _truncate_text(arch[k], max_arch_summary_len)

        out = {
            "id": ev.get("id"),
            "type": ev.get("type"),
            "confidence": ev.get("confidence"),
            "summary": ev.get("summary"),
            "detail": detail,
            "evidence": [],
        }

        # truncate evidence note
        for e in ev.get("evidence", []) or []:
            if not isinstance(e, dict):
                continue
            note = e.get("note", "") or ""
            if len(note) > max_evidence_note_len:
                note = note[: max_evidence_note_len - 3] + "..."
            out["evidence"].append({"kind": e.get("kind"), "ref": e.get("ref"), "note": note})

        changes_out.append(out)

    # meta: keep only what LLM needs for title
    meta_out = {
        "repo": meta.get("repo"),
        "version_a": meta.get("version_a"),
        "version_b": meta.get("version_b"),
        "generated_at": meta.get("generated_at"),
    }

    quality_out = quality
    entities_out = entities

    return {"meta": meta_out, "quality": quality_out, "entities": entities_out, "changes": changes_out}


def _group_changes(changes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {"Files": [], "Modules": [], "Components": [], "Quality": [], "Other": []}
    for ev in changes:
        t = ev.get("type")
        if t in FILE_TYPES:
            groups["Files"].append(ev)
        elif t in MODULE_TYPES:
            groups["Modules"].append(ev)
        elif t in COMP_TYPES:
            groups["Components"].append(ev)
        elif t in QUALITY_TYPES:
            groups["Quality"].append(ev)
        else:
            groups["Other"].append(ev)
    return groups


def _fmt_bullets(items: List[str], indent: str = "  ") -> str:
    if not items:
        return ""
    return "\n".join([f"{indent}- {x}" for x in items])


def _indent_block(block: str, prefix: str = "  ") -> str:
    if not block:
        return ""
    return "\n".join(prefix + line if line.strip() else line for line in block.splitlines())


def _render_semantics_block(change: Dict[str, Any]) -> str:
    """
    从 change['detail']['semantics'] 中提取可读文本块。
    - Architecture context：component 摘要 + patterns
    - Code semantics：added/removed files 的语义描述（path + desc）
    """
    detail = change.get("detail") or {}
    sem = detail.get("semantics") or {}
    if not isinstance(sem, dict):
        return ""

    code = sem.get("code") or {}
    arch = sem.get("arch") or {}

    lines: List[str] = []

    # --- Architecture context ---
    arch_lines: List[str] = []
    if isinstance(arch, dict):
        from_comp = arch.get("from_component")
        to_comp = arch.get("to_component")

        if from_comp or to_comp:
            if from_comp and to_comp and from_comp != to_comp:
                arch_lines.append(f"Component: `{from_comp}` → `{to_comp}`")
            elif from_comp:
                arch_lines.append(f"Component: `{from_comp}`")
            elif to_comp:
                arch_lines.append(f"Component: `{to_comp}`")

        from_sum = arch.get("from_component_summary")
        to_sum = arch.get("to_component_summary")
        if isinstance(from_sum, str) and from_sum.strip():
            arch_lines.append(f"From component semantics: {from_sum.strip()}")
        if isinstance(to_sum, str) and to_sum.strip():
            arch_lines.append(f"To component semantics: {to_sum.strip()}")

        pa = arch.get("patterns_a_top") or []
        pb = arch.get("patterns_b_top") or []
        if isinstance(pa, list) and pa:
            arch_lines.append("Arch patterns (source, top): " + ", ".join([f"`{p}`" for p in pa]))
        if isinstance(pb, list) and pb:
            arch_lines.append("Arch patterns (target, top): " + ", ".join([f"`{p}`" for p in pb]))

    if arch_lines:
        lines.append("**Architecture context**")
        lines.extend([f"- {x}" for x in arch_lines])

    # --- Code semantics ---
    def _entry_to_line(e: Dict[str, Any]) -> str:
        p = (e.get("path") or "").strip()
        d = (e.get("desc") or "").strip()
        if not p and not d:
            return ""
        if d:
            return f"`{p}`: {d}"
        return f"`{p}`"

    code_lines_added: List[str] = []
    code_lines_removed: List[str] = []
    if isinstance(code, dict):
        added_entries = code.get("added_files") or []
        removed_entries = code.get("removed_files") or []
        if isinstance(added_entries, list):
            code_lines_added = [_entry_to_line(e) for e in added_entries if isinstance(e, dict)]
            code_lines_added = [x for x in code_lines_added if x]
        if isinstance(removed_entries, list):
            code_lines_removed = [_entry_to_line(e) for e in removed_entries if isinstance(e, dict)]
            code_lines_removed = [x for x in code_lines_removed if x]

    if code_lines_added or code_lines_removed:
        lines.append("**Code semantics (evidence from CodeSem)**")
        if code_lines_added:
            lines.append("- Added/introduced:")
            lines.append(_fmt_bullets(code_lines_added, indent="  "))
        if code_lines_removed:
            lines.append("- Removed/retired:")
            lines.append(_fmt_bullets(code_lines_removed, indent="  "))

    if not lines:
        return ""

    return "\n".join(lines)


def render_markdown_template(ir: Dict[str, Any]) -> str:
    """
    模板法 Markdown（不依赖 LLM，稳定可测试）
    强制每条 bullet 引用 [CHG-XXXX]
    """
    meta = ir.get("meta", {})
    version_a = meta.get("version_a", "A")
    version_b = meta.get("version_b", "B")

    quality = ir.get("quality", {}) or {}
    entities = ir.get("entities", {}) or {}
    changes = ir.get("changes", []) or []

    groups = _group_changes([c for c in changes if isinstance(c, dict)])

    lines: List[str] = []
    lines.append(f"# Architecture Change Report: {version_a} → {version_b}")
    lines.append("")
    lines.append("## Overview")

    # overview bullets (2-4)
    files_ent = (entities.get("files") or {})
    fa = files_ent.get("count_a")
    fb = files_ent.get("count_b")
    added = files_ent.get("added") or []
    removed = files_ent.get("removed") or []

    # Find a stable_file_universe warning event id if available
    stable_ids = [c.get("id") for c in groups["Quality"] if "stable_file_universe" in str(c.get("detail", {}))]
    stable_id = stable_ids[0] if stable_ids else (groups["Quality"][0].get("id") if groups["Quality"] else None)

    if fa is not None and fb is not None:
        cite = stable_id or (changes[0].get("id") if changes else "CHG-0000")
        lines.append(f"- File universe: {fa} → {fb} (added={len(added)}, removed={len(removed)}). [{cite}]")

    if changes:
        cite = changes[0].get("id", "CHG-0000")
        lines.append(f"- Total detected change events: {len(changes)}. [{cite}]")

    caution_flags = []
    for k in ["module_count_delta_large", "component_mapping_incomplete", "namedcluster_has_duplicate_module_names"]:
        if quality.get(k) is True:
            caution_flags.append(k)
    if caution_flags:
        q_cite = (groups["Quality"][0].get("id") if groups["Quality"] else (changes[0].get("id") if changes else "CHG-0000"))
        lines.append(f"- Reliability caution due to flags: {caution_flags}. [{q_cite}]")

    lines.append("")
    lines.append("## Detected Changes")

    def emit_section(title: str, evs: List[Dict[str, Any]]) -> None:
        lines.append(f"### {title}")
        if not evs:
            if changes:
                lines.append(f"- No events in this category. [{changes[0].get('id','CHG-0000')}]")
            else:
                lines.append("- No events in this category.")
            lines.append("")
            return

        for ev in evs:
            cid = ev.get("id", "CHG-0000")
            conf = float(ev.get("confidence", 0.0) or 0.0)
            low = " (Low confidence)" if conf < 0.75 else ""
            summary = (ev.get("summary", "") or ev.get("type", "")).strip()
            lines.append(f"- {summary}{low}. [{cid}]")

            # Optional: show file examples for module_changed
            detail = ev.get("detail") or {}
            examples = detail.get("examples") or {}
            added_top = examples.get("added_files_top") or []
            removed_top = examples.get("removed_files_top") or []
            if isinstance(added_top, list) and added_top:
                lines.append("  - Added files (top): " + ", ".join([f"`{x}`" for x in added_top]))
            if isinstance(removed_top, list) and removed_top:
                lines.append("  - Removed files (top): " + ", ".join([f"`{x}`" for x in removed_top]))

            # NEW: semantics block (CodeSem/ArchSem evidence)
            sem_block = _render_semantics_block(ev)
            if sem_block:
                lines.append(_indent_block(sem_block, prefix="  "))

        lines.append("")

    emit_section("Files", groups["Files"])
    emit_section("Modules", groups["Modules"])
    emit_section("Components", groups["Components"])
    emit_section("Quality", groups["Quality"])

    lines.append("## Reliability notes")
    q_cite = (groups["Quality"][0].get("id") if groups["Quality"] else (changes[0].get("id") if changes else "CHG-0000"))
    notes = quality.get("notes") or []
    if notes:
        for n in notes:
            lines.append(f"- {n} [{q_cite}]")
    else:
        lines.append(f"- No additional reliability notes. [{q_cite}]")

    lines.append("")
    lines.append("## Appendix: Change Index")
    for ev in [c for c in changes if isinstance(c, dict)]:
        cid = ev.get("id", "CHG-0000")
        summary = (ev.get("summary", "") or ev.get("type", "")).strip()
        lines.append(f"- {cid}: {summary}")

    return "\n".join(lines)


def render_markdown_llm(ir: Dict[str, Any], model: str = "deepseek-chat") -> str:
    slim = slim_ir_for_llm(ir)
    return generate_markdown_from_ir(slim, model=model)


def main() -> None:
    """
    右键运行自测：
    - 填写 diff_ir.json 路径
    - 输出 diff_summary.template.md（模板法）
    - 如果配置了 DEEPSEEK_API_KEY，可输出 diff_summary.llm.md
    """
    ir_path = Path(r"out\out_demo_ir.json")  # TODO: 改成你的实际路径
    if not ir_path.exists():
        raise FileNotFoundError(f"diff_ir.json not found: {ir_path}")

    ir = load_ir_json(ir_path)

    md1 = render_markdown_template(ir)
    out1 = ir_path.parent / "diff_summary.template.md"
    out1.write_text(md1, encoding="utf-8")
    print(f"Wrote: {out1}")

    # optional llm
    try:
        md2 = render_markdown_llm(ir, model="deepseek-chat")
        out2 = ir_path.parent / "diff_summary.llm.md"
        out2.write_text(md2, encoding="utf-8")
        print(f"Wrote: {out2}")
    except Exception as e:
        print(f"LLM summary skipped: {e}")


if __name__ == "__main__":
    main()
