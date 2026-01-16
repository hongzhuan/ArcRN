"""
render_md.py
- 从 DiffIR（或 diff_ir.json）生成 Markdown
- 支持两种模式：
  1) template：不使用 LLM，稳定输出
  2) llm：调用 deepseek-chat，生成更自然的报告

Stage-2（当前策略）：
- LLM 输入来自 diff_ir-summary.json
- 只喂 meta（可选）、quality（可选）、entities（可选）
- changes 只喂 id / type / summary
- 不喂 files desc，不喂 detail.semantics，不喂 ArchSem/CodeSem 长文本
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


# Stage-2 专用：更严格的 system prompt（只允许用 summary）
SYSTEM_PROMPT_STAGE2 = """你是“软件架构变更报告”生成助手。你的回答必须使用中文，并且严格基于输入的 diff_ir-summary.json 内容。

========================
一、硬性规则（必须遵守）
========================
1) 严禁编造事实：只能使用 changes[].summary 中明确给出的信息。禁止补充背景、原因、动机、影响或任何推测性内容。
2) 禁止使用未提供字段，请不要提及这些字段或从中推断信息。
3) 强制引用 Change ID：凡是描述“发生了什么变更”的句子，必须在句末引用至少一个 Change ID，格式为 [CHG-XXXX]。
   - 如果无法为一句话找到对应的 Change ID，则不要写这句话。
4) 置信度标注：如果输入里没有 confidence 字段，请不要自行判断“低置信度”。（本阶段仅基于 summary 编排内容）
5) 输出必须是合法的 Markdown，语言简洁、技术化，不要输出 JSON 原文或长段代码块。

========================
二、写作目标
========================
你的目标是生成一份“面向架构评审与版本对比”的报告，重点说明：
- 模块层面的新增 / 删除 / 变更概况
- 每条变更的 summary（来自 Stage-1）如何归纳进报告结构
而不是解释“为什么发生”或“会带来什么业务后果”。

========================
三、输出结构（必须严格遵守）
========================
# Architecture Change Report: <version_a> → <version_b>

## Overview
- 2–4 条要点，总结整体变化范围（如变更事件数量、模块增删改数量等）。
- 每一条都必须引用至少一个 Change ID，例如 [CHG-0001]。

## Detected Changes
按以下子类分组，每一条都必须引用 Change ID：
### 新增模块

### 删除模块

### 变更模块


## Appendix: Change Index
- 按顺序列出所有 Change：
  - CHG-XXXX: <summary>
"""


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
    旧版精简器（template 或旧策略可能仍会用到）：
    - 保留 detail/semantics，并做截断
    当前 Stage-2 的 llm 路径不会调用该函数，但保留以兼容历史用法。
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

    meta_out = {
        "repo": meta.get("repo"),
        "version_a": meta.get("version_a"),
        "version_b": meta.get("version_b"),
        "generated_at": meta.get("generated_at"),
    }

    return {"meta": meta_out, "quality": quality, "entities": entities, "changes": changes_out}


def slim_ir_for_stage2(ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage-2 专用精简器：
    只保留 meta（必要字段）、quality（可选）、entities（可选）、
    changes 中只保留 id / type / summary / detail.module_name。
    """
    meta = ir.get("meta", {}) or {}
    out: Dict[str, Any] = {
        "meta": {
            "repo": meta.get("repo"),
            "version_a": meta.get("version_a"),
            "version_b": meta.get("version_b"),
            "generated_at": meta.get("generated_at"),
        }
    }

    # quality/entities 为可选：按你要求“如果需要的话”
    if "quality" in ir:
        out["quality"] = ir.get("quality") or {}
    if "entities" in ir:
        out["entities"] = ir.get("entities") or {}

    changes_out: List[Dict[str, Any]] = []
    for ev in ir.get("changes", []) or []:
        if not isinstance(ev, dict):
            continue
        changes_out.append({
            "id": ev.get("id"),
            "type": ev.get("type"),
            "summary": ev.get("summary"),
            "module_name": ev.get("detail").get("module_name")
        })

    out["changes"] = changes_out
    return out


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
    （template 模式使用）从 change['detail']['semantics'] 中提取可读文本块。
    Stage-2 LLM 不会喂 semantics，因此 llm 模式不再依赖此函数。
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

            detail = ev.get("detail") or {}
            examples = detail.get("examples") or {}
            added_top = examples.get("added_files_top") or []
            removed_top = examples.get("removed_files_top") or []
            if isinstance(added_top, list) and added_top:
                lines.append("  - Added files (top): " + ", ".join([f"`{x}`" for x in added_top]))
            if isinstance(removed_top, list) and removed_top:
                lines.append("  - Removed files (top): " + ", ".join([f"`{x}`" for x in removed_top]))

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
    """
    Stage-2：基于 diff_ir-summary.json（每条 change 的 summary 已由 Stage-1 生成）
    只喂 meta/quality/entities + changes(id/type/summary)
    """
    slim = slim_ir_for_stage2(ir)
    return generate_markdown_from_ir(
        slim,
        model=model,
        system_prompt=SYSTEM_PROMPT_STAGE2,
    )


def main() -> None:
    """
    右键运行自测：
    - 填写 diff_ir-summary.json 路径
    - 输出 diff_summary.template.md（模板法）
    - 如果配置了 DEEPSEEK_API_KEY，可输出 diff_summary.llm.md（Stage-2）
    """
    ir_path = Path(r"out\diff_ir-summary.json")  #
    if not ir_path.exists():
        raise FileNotFoundError(f"diff_ir.json not found: {ir_path}")

    ir = load_ir_json(ir_path)

    md1 = render_markdown_template(ir)
    out1 = ir_path.parent / "diff_summary.template.md"
    out1.write_text(md1, encoding="utf-8")
    print(f"Wrote: {out1}")

    # optional llm (Stage-2)
    try:
        md2 = render_markdown_llm(ir, model="deepseek-chat")
        out2 = ir_path.parent / "diff_summary.llm.md"
        out2.write_text(md2, encoding="utf-8")
        print(f"Wrote: {out2}")
    except Exception as e:
        print(f"LLM summary skipped: {e}")


if __name__ == "__main__":
    main()
