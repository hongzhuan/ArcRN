"""
run_diff.py (module-level)
- PyCharm 右键运行入口（不依赖终端参数）
- 输入：两个目录（版本A/版本B），每个目录中是该版本的 SemArc JSON 输出
- 核心：使用 a2a_jaccard 进行模块对齐，然后输出模块级 diff_ir.json（不再逐文件刷屏）
"""

from __future__ import annotations

import os
import json
import copy
from pathlib import Path

from sema_diff.config import DiffConfig, default_config
from sema_diff.loader import resolve_inputs_from_dirs, ResolvedInputs

from sema_diff.parse_namedclusters import parse_namedclusters, NamedClustersIndex
from sema_diff.parse_clustercomponent import parse_clustercomponent, ComponentMapping

from sema_diff.a2a_jaccard import build_module_files, align_modules_by_jaccard
from sema_diff.module_diff_core import build_module_level_events

from sema_diff.quality import build_quality_report
from sema_diff.diff_core import build_snapshot, diff_file_universe  # 仅用于质量与实体统计
from sema_diff.ir import DiffIR, now_iso_local, dump_ir

from sema_diff.render_md import render_markdown_template, render_markdown_llm

from sema_diff.parse_codesem import parse_codesem, CodeSemIndex
from sema_diff.parse_archsem import parse_archsem, ArchSemIndex

from sema_diff.denoise import denoise_changes

def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # TODO: 改成你的实际路径（PyCharm 里直接改变量即可）
    dir_a = Path(r"sema_results/libxml2-v2.14.2")
    dir_b = Path(r"sema_results/libxml2-v2.14.3")

    version_a_label = "v2.14.2"
    version_b_label = "v2.14.3"
    repo_name = "libxml2"

    # === 模块级参数（降噪关键） ===
    min_edge_weight = 0.0          # jaccard 边过滤；模块多时可调到 0.05~0.10
    min_file_delta = 2             # 小变化不输出 module_changed（建议从 2/3 开始调）
    top_k_files = 8                # 每个模块变化最多展示几个新增/删除文件例子
    min_jaccard_to_accept = 0.0    # 过滤低质量 mapping（可调到 0.1/0.2）

    # === 可选：生成 Markdown Summary ===
    generate_md = True
    md_mode = "template"  # "template" or "llm"
    llm_model = "deepseek-chat"

    out_dir = Path(os.getcwd()) / "out" / f"{repo_name}_{version_a_label}-{version_b_label}"
    _ensure_out_dir(out_dir)

    cfg: DiffConfig = default_config()

    # 1) loader：定位输入文件
    resolved: ResolvedInputs = resolve_inputs_from_dirs(dir_a=dir_a, dir_b=dir_b)

    a_named_path = resolved.version_a_files["NamedClusters"]
    a_comp_path = resolved.version_a_files["ClusterComponent"]
    b_named_path = resolved.version_b_files["NamedClusters"]
    b_comp_path = resolved.version_b_files["ClusterComponent"]

    a_arch_path = resolved.version_a_files.get("ArchSem")
    a_code_path = resolved.version_a_files.get("CodeSem")
    b_arch_path = resolved.version_b_files.get("ArchSem")
    b_code_path = resolved.version_b_files.get("CodeSem")

    print("=== Inputs ===")
    print("A NamedClusters:", a_named_path)
    print("A ClusterComponent:", a_comp_path)
    print("B NamedClusters:", b_named_path)
    print("B ClusterComponent:", b_comp_path)

    # 2) parse
    idx_a: NamedClustersIndex = parse_namedclusters(a_named_path, cfg)
    idx_b: NamedClustersIndex = parse_namedclusters(b_named_path, cfg)

    comp_a: ComponentMapping = parse_clustercomponent(a_comp_path, idx_a.name_to_uids_queue, cfg)
    comp_b: ComponentMapping = parse_clustercomponent(b_comp_path, idx_b.name_to_uids_queue, cfg)

    codesem_a = parse_codesem(a_code_path) if a_code_path and a_code_path.exists() else None
    codesem_b = parse_codesem(b_code_path) if b_code_path and b_code_path.exists() else None

    archsem_a = parse_archsem(a_arch_path) if a_arch_path and a_arch_path.exists() else None
    archsem_b = parse_archsem(b_arch_path) if b_arch_path and b_arch_path.exists() else None

    # 3) a2a_jaccard alignment (module mapping)
    modules_a = build_module_files(idx_a)
    modules_b = build_module_files(idx_b)

    alignment = align_modules_by_jaccard(
        modules_a,
        modules_b,
        min_edge_weight=min_edge_weight,
        engine="auto",
    )

    print("\n=== A2A Mapping Summary ===")
    print("engine:", alignment.meta.get("engine"))
    print("global_similarity:", alignment.global_similarity)
    print("mapped:", len(alignment.mapping), "removed(A-only):", len(alignment.removed), "added(B-only):", len(alignment.added))

    # 4) module-level events
    events = []
    next_id = 1

    mod_events, next_id = build_module_level_events(
        idx_a=idx_a,
        idx_b=idx_b,
        comp_a=comp_a,
        comp_b=comp_b,
        alignment=alignment,
        next_id_start=next_id,
        top_k_files=top_k_files,
        min_file_delta=min_file_delta,
        min_jaccard_to_accept=min_jaccard_to_accept,
        codesem_a=codesem_a,
        codesem_b=codesem_b,
        archsem_a=archsem_a,
        archsem_b=archsem_b,
    )

    events.extend(mod_events)

    # 5) quality（沿用旧质量框架：文件全集稳定性、重复模块名、组件映射不完整等）
    # 为了复用 quality.py，这里构建 snapshot 并计算 file universe diff（仅用于质量画像，不输出 file 事件）
    snap_a = build_snapshot(version_a_label, idx_a, comp_a)
    snap_b = build_snapshot(version_b_label, idx_b, comp_b)
    files_added, files_removed, reassigned = diff_file_universe(snap_a, snap_b)

    quality_report, next_id = build_quality_report(
        snap_a,
        snap_b,
        files_added_count=len(files_added),
        files_removed_count=len(files_removed),
        cfg=cfg,
        next_id_start=next_id,
    )
    # 质量事件保留（一般不多，且对解释很有用）
    events.extend(quality_report.warning_events)

    # 6) assemble IR（模块级）
    meta = {
        "repo": repo_name,
        "version_a": version_a_label,
        "version_b": version_b_label,
        "generated_at": now_iso_local(),
        "inputs": {
            "named_clusters_a": str(a_named_path),
            "named_clusters_b": str(b_named_path),
            "cluster_component_a": str(a_comp_path),
            "cluster_component_b": str(b_comp_path),
        },
        "a2a": {
            "engine": alignment.meta.get("engine"),
            "global_similarity": alignment.global_similarity,
            "min_edge_weight": min_edge_weight,
        },
        "module_diff": {
            "min_file_delta": min_file_delta,
            "top_k_files": top_k_files,
            "min_jaccard_to_accept": min_jaccard_to_accept,
        },
        "semantics": {
            "codesem_a_loaded": bool(codesem_a),
            "codesem_b_loaded": bool(codesem_b),
            "archsem_a_loaded": bool(archsem_a),
            "archsem_b_loaded": bool(archsem_b),
            "codesem_a_size": (len(codesem_a.file_to_desc) if codesem_a else 0),
            "codesem_b_size": (len(codesem_b.file_to_desc) if codesem_b else 0),
            "archsem_a_components": (len(archsem_a.component_to_summary) if archsem_a else 0),
            "archsem_b_components": (len(archsem_b.component_to_summary) if archsem_b else 0),
        },
    }

    quality = {**quality_report.flags, "notes": quality_report.notes}

    entities = {
        "files": {
            "count_a": len(snap_a.files),
            "count_b": len(snap_b.files),
            "added": sorted(files_added),
            "removed": sorted(files_removed),
            "reassigned_count": len(reassigned),
        },
        "modules": {
            "count_a": len(idx_a.modules),
            "count_b": len(idx_b.modules),
            "mapped": len(alignment.mapping),
            "removed": len(alignment.removed),
            "added": len(alignment.added),
        },
        "components": {
            "count_a": len(comp_a.component_to_module_uids),
            "count_b": len(comp_b.component_to_module_uids),
            "unresolved_a": len(comp_a.unresolved),
            "unresolved_b": len(comp_b.unresolved),
        },
    }

    ir = DiffIR(meta=meta, quality=quality, entities=entities, changes=events)

    # === 6.1 写出 raw IR（未降噪，供对照/回溯） ===
    raw_path = out_dir / "diff_ir-raw.json"
    dump_ir(ir, str(raw_path), pretty=True)
    print(f"\nWrote RAW IR: {raw_path}")
    print(f"RAW total changes: {len(events)}")

    # === 6.2 生成 denoised IR（Step-1：过滤 rename 噪声） ===
    ir_denoised = copy.deepcopy(ir)
    filtered_changes, denoise_stats = denoise_changes(
        changes=ir_denoised.changes,
        named_a=idx_a,
        named_b=idx_b,
        cfg=cfg.denoise,
    )
    ir_denoised.changes = filtered_changes
    # 把降噪统计写入 meta，便于实验记录
    ir_denoised.meta["denoise"] = denoise_stats

    denoised_path = out_dir / "diff_ir-denoised.json"
    dump_ir(ir_denoised, str(denoised_path), pretty=True)
    print(f"Wrote DENOISED IR: {denoised_path}")
    print(f"DENOISED total changes: {len(filtered_changes)} (dropped={denoise_stats.get('dropped')})")

    # === 7) optional markdown summary（默认用 denoised 作为输入） ===
    if generate_md:
        ir_dict = json.loads(denoised_path.read_text(encoding="utf-8"))

        if md_mode == "template":
            md = render_markdown_template(ir_dict)
        elif md_mode == "llm":
            md = render_markdown_llm(ir_dict, model=llm_model)
        else:
            raise ValueError(f"Unknown md_mode: {md_mode}")

        md_path = out_dir / "diff_summary.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"Wrote Markdown summary: {md_path} (mode={md_mode})")

    print("Done.")


if __name__ == "__main__":
    main()
