"""
config.py
- 统一管理阈值与策略开关
- MVP 只需要这些阈值即可跑通
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffConfig:
    # 模块“重命名/等价”判定：overlap >= rename_overlap
    rename_overlap: float = 0.90

    # 拆分/合并候选：overlap >= split_merge_overlap
    split_merge_overlap: float = 0.30

    # 拆分/合并覆盖率阈值：coverage >= coverage_threshold
    coverage_threshold: float = 0.80

    # 触发质量告警：模块数量变化比例（例如 14->9 变化 35.7%）
    module_count_delta_warn_ratio: float = 0.30

    # 是否启用 occurrence 消歧（ClusterComponent 里模块名重复时）
    enable_occurrence_disambiguation: bool = True

    # 文件路径归一化：是否将 '\' 替换为 '/'
    normalize_path_separators: bool = True


def default_config() -> DiffConfig:
    return DiffConfig()


def main() -> None:
    cfg = default_config()
    print("Default DiffConfig:")
    print(cfg)


if __name__ == "__main__":
    main()
