"""
parse_codesem.py
- 解析 SemArc 输出的 *_CodeSem.json
- 目标：构建 file_path -> description 的索引（尽量适配不同 schema）
- 只做“语义索引”，不做 diff

策略（尽量鲁棒）：
- 递归扫描 JSON（dict/list）
- 识别“看起来像文件路径”的字段 + “看起来像描述”的字段
- 路径统一归一化（\\ -> /，去掉 ./）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_path(p: str) -> str:
    p = p.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def _looks_like_file_path(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) < 3 or len(s) > 300:
        return False

    # 排除 URL
    lower = s.lower()
    if lower.startswith(("http://", "https://")):
        return False

    # 允许“纯文件名”（如 shell.c / xmllint.c），也允许带目录
    exts = (
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".m", ".mm",
        ".py", ".js", ".ts", ".java", ".kt", ".go", ".rs", ".cs", ".swift",
        ".md", ".txt", ".json", ".yml", ".yaml"
    )
    return any(lower.endswith(e) for e in exts)


def _choose_desc_from_dict(d: Dict[str, Any]) -> Optional[str]:
    """
    从一个 dict 中尽量挑出“描述文本”。
    适配：libxml2 CodeSem 使用 "Functionality"。
    """
    # NEW: 直接支持 libxml2 schema
    for k in ("Functionality", "functionality"):
        v = d.get(k)
        if isinstance(v, str):
            t = v.strip()
            if len(t) >= 5:
                return t

    # 常见候选字段（兼容其它项目）
    candidates = [
        "description", "desc", "summary", "semantics", "semantic", "meaning",
        "function", "purpose", "responsibility", "comment", "explain", "explanation",
        "content"
    ]
    for k in candidates:
        v = d.get(k)
        if isinstance(v, str):
            t = v.strip()
            if len(t) >= 5:
                return t

    # 最后兜底：挑一个“足够长”的字符串字段（排除明显不是描述的 key）
    for k, v in d.items():
        if not isinstance(v, str):
            continue
        if k.lower() in ("path", "file", "name", "filename", "fullpath", "relative_path"):
            continue
        t = v.strip()
        if len(t) >= 10:
            return t

    return None


def _extract_file_desc_pairs(obj: Any) -> List[Tuple[str, str]]:
    """
    递归扫描：返回一组 (file_path, desc)。
    """
    pairs: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        # 先在当前 dict 层尝试直接匹配
        file_keys = ["file", "path", "filepath", "file_path", "filename", "name", "fullpath", "relative_path"]
        path_val: Optional[str] = None
        for fk in file_keys:
            v = obj.get(fk)
            if isinstance(v, str) and _looks_like_file_path(v):
                path_val = v
                break

        if path_val is not None:
            desc = _choose_desc_from_dict(obj)
            if desc:
                pairs.append((normalize_path(path_val), desc))

        # 再递归子节点
        for v in obj.values():
            pairs.extend(_extract_file_desc_pairs(v))

    elif isinstance(obj, list):
        for item in obj:
            pairs.extend(_extract_file_desc_pairs(item))

    return pairs


@dataclass(frozen=True)
class CodeSemIndex:
    file_to_desc: Dict[str, str]
    total_pairs_found: int
    source_path: str


def parse_codesem(json_path: Path) -> CodeSemIndex:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    pairs: List[Tuple[str, str]] = []

    # NEW: fast path for {"summary": [ {file, Functionality}, ... ]}
    if isinstance(data, dict) and isinstance(data.get("summary"), list):
        for it in data["summary"]:
            if not isinstance(it, dict):
                continue
            fp = it.get("file")
            if isinstance(fp, str) and _looks_like_file_path(fp):
                desc = _choose_desc_from_dict(it)
                if desc:
                    pairs.append((normalize_path(fp), desc))
    else:
        # fallback: recursive extraction for other schemas
        pairs = _extract_file_desc_pairs(data)

    file_to_desc: Dict[str, str] = {}
    for fp, desc in pairs:
        # 同一个文件可能被多次提取；保留更长的描述（信息量更大）
        if fp not in file_to_desc or len(desc) > len(file_to_desc[fp]):
            file_to_desc[fp] = desc

    return CodeSemIndex(
        file_to_desc=file_to_desc,
        total_pairs_found=len(pairs),
        source_path=str(json_path),
    )


def main() -> None:
    # TODO: 改成你的路径
    p = Path(r"C:\path\to\libuv-v1.49.1_CodeSem.json")
    if not p.exists():
        print(f"Not found: {p}")
        return

    idx = parse_codesem(p)
    print("CodeSem parsed.")
    print("source:", idx.source_path)
    print("pairs found:", idx.total_pairs_found)
    print("unique files:", len(idx.file_to_desc))

    # 打印前 5 条
    for i, (k, v) in enumerate(list(idx.file_to_desc.items())[:5]):
        print(f"\n[{i}] {k}\n{v[:200]}{'...' if len(v) > 200 else ''}")


if __name__ == "__main__":
    main()
