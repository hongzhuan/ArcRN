"""
parse_namedclusters.py
- 解析 SemArc 输出的 *_NamedClusters.json
- 输出：
  1) modules: List[Module]
  2) file_to_module_uid: Dict[file_path, module_uid]
  3) name_to_uids_queue: Dict[module_name, deque([module_uid...])]  # 给 ClusterComponent occurrence 消歧用
  4) quality info: duplicate module names, empty modules
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque

from sema_diff.config import DiffConfig, default_config


@dataclass(frozen=True)
class Module:
    uid: str                 # e.g., "libuv#1"
    name: str                # e.g., "libuv"
    occurrence: int          # 1-based
    files: Set[str]          # normalized relative paths
    file_count: int
    signature: str           # sha1 of sorted files (short)


@dataclass(frozen=True)
class NamedClustersIndex:
    modules: List[Module]
    file_to_module_uid: Dict[str, str]               # file -> module_uid
    name_to_uids_queue: Dict[str, "deque[str]"]      # name -> deque([uid1, uid2, ...])

    duplicate_module_names: List[str]
    empty_modules: List[str]                         # list of module_uid with 0 files

    raw_name: Optional[str] = None                   # JSON top-level 'name'
    schema_version: Optional[str] = None             # JSON top-level '@schemaVersion'


def _norm_path(p: str, cfg: DiffConfig) -> str:
    s = p.strip()
    if cfg.normalize_path_separators:
        s = s.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    # collapse duplicate slashes
    while "//" in s:
        s = s.replace("//", "/")
    return s


def _module_signature(files: Set[str]) -> str:
    joined = "\n".join(sorted(files)).encode("utf-8", errors="ignore")
    return sha1(joined).hexdigest()[:10]


def parse_namedclusters(json_path: Path, cfg: DiffConfig) -> NamedClustersIndex:
    if not json_path.exists():
        raise FileNotFoundError(f"NamedClusters json not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    schema_version = data.get("@schemaVersion")
    raw_name = data.get("name")

    structure = data.get("structure")
    if not isinstance(structure, list):
        raise ValueError("Invalid NamedClusters.json: top-level 'structure' must be a list.")

    occurrence_counter: Dict[str, int] = defaultdict(int)
    modules: List[Module] = []
    file_to_module_uid: Dict[str, str] = {}

    for node in structure:
        if not isinstance(node, dict):
            continue
        if node.get("@type") != "group":
            continue

        name = str(node.get("name", "")).strip()
        occurrence_counter[name] += 1
        occ = occurrence_counter[name]
        uid = f"{name}#{occ}"

        nested = node.get("nested", [])
        files: Set[str] = set()

        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    continue
                if item.get("@type") != "item":
                    continue
                f = item.get("name")
                if not isinstance(f, str):
                    continue
                nf = _norm_path(f, cfg)
                if nf:
                    files.add(nf)

        sig = _module_signature(files)
        mod = Module(
            uid=uid,
            name=name,
            occurrence=occ,
            files=files,
            file_count=len(files),
            signature=sig,
        )
        modules.append(mod)

        # file -> module mapping (if a file appears in multiple modules, keep the first and warn later)
        for f in files:
            if f not in file_to_module_uid:
                file_to_module_uid[f] = uid

    # build name -> uids queue
    name_to_uids_queue: Dict[str, "deque[str]"] = {}
    for mod in modules:
        name_to_uids_queue.setdefault(mod.name, deque()).append(mod.uid)

    # quality: duplicate names, empty modules
    duplicate_module_names = sorted([n for n, c in occurrence_counter.items() if c > 1])
    empty_modules = [m.uid for m in modules if m.file_count == 0]

    return NamedClustersIndex(
        modules=modules,
        file_to_module_uid=file_to_module_uid,
        name_to_uids_queue=name_to_uids_queue,
        duplicate_module_names=duplicate_module_names,
        empty_modules=empty_modules,
        raw_name=raw_name,
        schema_version=schema_version,
    )


def main() -> None:
    """
    右键运行本文件进行自测：
    - 你只需要填写 json_path（或填写 dir 然后自动找 *_NamedClusters.json）
    """
    cfg = default_config()

    # TODO: 方式1：直接填 NamedClusters.json 的完整路径
    json_path = Path(r"..\sema_results\libuv-1.49.0")

    # 方式2：只填目录，则尝试自动定位（建议你先用 loader 打印确认后再写死路径）
    if json_path.is_dir():
        candidates = list(json_path.rglob(f"*{'_NamedClusters.json'}"))
        if not candidates:
            raise RuntimeError(f"No *_NamedClusters.json found under {json_path}")
        candidates.sort(key=lambda p: (len(str(p)), str(p)))
        json_path = candidates[0]

    idx = parse_namedclusters(json_path, cfg)

    print("=== parse_namedclusters.main() ===")
    print(f"File: {json_path}")
    print(f"Schema: {idx.schema_version}, Name: {idx.raw_name}")
    print(f"Modules: {len(idx.modules)}")
    print(f"Files mapped: {len(idx.file_to_module_uid)}")
    print(f"Duplicate module names: {idx.duplicate_module_names}")
    print(f"Empty modules: {idx.empty_modules}")

    # show first 5 modules
    print("\nFirst 5 modules:")
    for m in idx.modules[:5]:
        print(f"  {m.uid} name={m.name} files={m.file_count} sig={m.signature}")


if __name__ == "__main__":
    main()
