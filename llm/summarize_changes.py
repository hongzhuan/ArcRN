# llm/summarize_changes.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .deepseek_client import generate_change_summary_structured


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_module_name_and_file_count(ev: Dict[str, Any]) -> Tuple[str, int]:
    """
    兼容不同事件类型的 detail 字段：
    - module_added/module_removed: detail.module_name + detail.file_count
    - module_changed: detail.from_name/to_name + counts.file_count_b 或 counts.file_count_a
    """
    detail = ev.get("detail") or {}
    t = (ev.get("type") or "").strip()

    if t in ("module_added", "module_removed"):
        mn = (detail.get("module_name") or "").strip()
        fc = _safe_int(detail.get("file_count"), 0)
        return mn, fc

    if t == "module_changed":
        to_name = (detail.get("to_name") or "").strip()
        from_name = (detail.get("from_name") or "").strip()
        mn = to_name or from_name or ""
        counts = detail.get("counts") or {}
        fc = _safe_int(counts.get("file_count_b"), _safe_int(counts.get("file_count_a"), 0))
        return mn, fc

    return "", 0


def _extract_semantics_code(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    只取 detail.semantics.code；Stage-1 不关心 arch/component。
    """
    detail = ev.get("detail") or {}
    sem = detail.get("semantics") or {}
    if not isinstance(sem, dict):
        return {"added_files": [], "removed_files": []}
    code = sem.get("code") or {}
    if not isinstance(code, dict):
        return {"added_files": [], "removed_files": []}

    def _norm(arr: Any):
        out = []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    out.append({"path": it.get("path", ""), "desc": it.get("desc", "")})
        return out

    return {
        "added_files": _norm(code.get("added_files") or []),
        "removed_files": _norm(code.get("removed_files") or []),
    }


def summarize_ir_changes(
    ir: Dict[str, Any],
    out_path: Path,
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> Dict[str, Any]:
    """
    Stage-1：逐条调用 LLM 生成纯文本摘要，并直接覆盖 change["summary"]。
    - 不保留原 summary
    - 单条失败：summary = "大模型调用失败"
    - 输出写入 out_path
    """
    meta = ir.get("meta") or {}
    repo = (meta.get("repo") or "").strip()
    version_a = (meta.get("version_a") or "").strip()
    version_b = (meta.get("version_b") or "").strip()

    changes = ir.get("changes") or []
    if not isinstance(changes, list):
        raise ValueError("Invalid IR: changes must be a list")

    out_ir = ir  # 就地更新
    out_ir.setdefault("meta", {})
    out_ir["meta"].setdefault("stage1_summary", {})
    out_ir["meta"]["stage1_summary"].update({
        "enabled": True,
        "output_file": str(out_path),
        "model": model,
        "api_key_env": api_key_env,
        "strategy": "replace_summary_with_llm_text",
    })

    summarized = 0
    failed = 0

    for ev in changes:
        if not isinstance(ev, dict):
            continue

        change_type = (ev.get("type") or "").strip()
        if change_type not in ("module_added", "module_removed", "module_changed"):
            continue

        module_name, file_count = _extract_module_name_and_file_count(ev)
        semantics_code = _extract_semantics_code(ev)

        try:
            text_summary = generate_change_summary_structured(
                repo=repo,
                version_a=version_a,
                version_b=version_b,
                change_type=change_type,
                module_name=module_name,
                file_count=file_count,
                semantics_code=semantics_code,
                model=model,
                api_key_env=api_key_env,
            )
            # 直接覆盖 summary（不保留原 summary）
            ev["summary"] = (text_summary or "").strip() or "大模型调用失败"
            summarized += 1
        except Exception:
            ev["summary"] = "大模型调用失败"
            failed += 1

    out_ir["meta"]["stage1_summary"].update({
        "summarized_events": summarized,
        "failed_events": failed,
    })

    out_path.write_text(json.dumps(out_ir, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_ir


def summarize_changes_file(
    in_path: Path,
    out_path: Path,
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> Dict[str, Any]:
    ir = json.loads(in_path.read_text(encoding="utf-8"))
    return summarize_ir_changes(ir=ir, out_path=out_path, model=model, api_key_env=api_key_env)


def main() -> None:
    in_path = Path(r"out/diff_ir-denoised.json")
    out_path = in_path.parent / "diff_ir-summary.json"
    summarize_changes_file(in_path=in_path, out_path=out_path, model="deepseek-chat")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
