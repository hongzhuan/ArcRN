"""
ir.py
- IR 数据结构 + JSON 导出
- MVP：meta / quality / entities / changes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    kind: str     # e.g., "NamedClusters", "ClusterComponent", "Derived"
    ref: str      # e.g., "module:libuv#1", "file:src/uv-common.c"
    note: str = ""


@dataclass
class ChangeEvent:
    id: str
    type: str
    confidence: float
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class DiffIR:
    meta: Dict[str, Any]
    quality: Dict[str, Any]
    entities: Dict[str, Any]
    changes: List[ChangeEvent]


def now_iso_local() -> str:
    # 以本机时区输出（Windows 下通常可用）
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _to_jsonable(obj: Any) -> Any:
    # 将 dataclass/自定义对象转换为可 JSON 序列化的 dict
    if isinstance(obj, DiffIR):
        d = {
            "meta": obj.meta,
            "quality": obj.quality,
            "entities": obj.entities,
            "changes": [_to_jsonable(ev) for ev in obj.changes],
        }
        return d
    if isinstance(obj, ChangeEvent):
        return {
            "id": obj.id,
            "type": obj.type,
            "confidence": round(float(obj.confidence), 4),
            "summary": obj.summary,
            "detail": obj.detail,
            "evidence": [_to_jsonable(e) for e in obj.evidence],
        }
    if isinstance(obj, EvidenceItem):
        return {"kind": obj.kind, "ref": obj.ref, "note": obj.note}
    return obj


def dump_ir(ir: DiffIR, out_path: str, pretty: bool = True) -> None:
    payload = _to_jsonable(ir)
    if pretty:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    # 自测：写一个最小 IR
    ir = DiffIR(
        meta={"repo": "demo", "version_a": "A", "version_b": "B", "generated_at": now_iso_local()},
        quality={"stable_file_universe": True, "notes": []},
        entities={"files": {"count_a": 0, "count_b": 0, "added": [], "removed": []}},
        changes=[
            ChangeEvent(
                id="CHG-0001",
                type="file_added",
                confidence=1.0,
                summary="Added file demo.txt.",
                detail={"file": "demo.txt"},
                evidence=[EvidenceItem(kind="NamedClusters", ref="file:demo.txt", note="Example")],
            )
        ],
    )
    dump_ir(ir, r"out\out_demo_ir.json", pretty=True)
    print("Wrote: out_demo_ir.json")


if __name__ == "__main__":
    main()
