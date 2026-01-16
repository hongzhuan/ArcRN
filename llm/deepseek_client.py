"""
deepseek_client.py
- DeepSeek (OpenAI SDK) 调用封装
- 强约束：不幻觉 + 每条变更必须引用 [CHG-XXXX]
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# Stage 2，将多个 change summary 组合成最终报告的 prompt
SYSTEM_PROMPT_STRICT = """你是“软件架构变更报告”生成助手。你的回答必须使用中文，并且严格基于输入的 diff_ir-summary.json 内容。

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


def _get_client(api_key_env: str = "DEEPSEEK_API_KEY") -> OpenAI:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing environment variable {api_key_env}. "
            f"Please set it before running LLM summary."
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def generate_markdown_from_ir(
        ir_payload: Dict[str, Any],
        model: str = DEFAULT_MODEL,
        system_prompt: str = SYSTEM_PROMPT_STRICT,
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.2,
        max_tokens: int = 1200,
) -> str:
    """
    输入：已“精简后的”IR dict（建议只包含 meta/quality/entities/changes 的必要字段）
    输出：Markdown 文本
    """
    client = _get_client(api_key_env=api_key_env)

    user_prompt = (
        "请严格基于下面给出的 diff_ir.json 内容，生成一份 Markdown 格式的软件架构变更报告。\n"
        "除 diff_ir.json 中明确给出的信息外，不得引入任何额外事实、推断或解释。\n\n"
        "diff_ir.json 内容如下：\n"
        f"{json.dumps(ir_payload, ensure_ascii=False, indent=2)}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

# === 为diff_ir-denoised.json生成summary，保存到新文件diff_ir-summary.json中 ===

CHANGE_SUMMARY_SYSTEM_PROMPT_ZH = """你是“架构变更事件摘要”生成助手。你必须使用中文，并严格基于输入内容生成摘要。

                                    硬性规则（必须遵守）：
                                    1) 严禁编造：只能使用输入里提供的事实；不得推断代码行为、功能效果、修复内容、性能影响等，除非 desc 明确表达。
                                    2) 只能基于本条 change 的字段生成摘要，禁止引用或联想其他 change。
                                    3) 输出只需要两段纯文本的话，不需要类似于“下面是我的回答”等这样的话，输出并符合指定 schema。
                                    4) 语气中立、技术化，避免夸张表述。
                                    5) 如果信息不足，请在对应字段返回空数组，并在 notes 中说明“证据不足”。
                                    
                                    输出 text schema（必须完全一致）：
                                    1.功能性变更说明：...（一两句话说明即可）
                                    2.非功能性变更说明:...（一两句话说明即可）
                                    
                                    功能性/非功能性划分指导（必须遵守）：
                                    - 只能根据输入的 file desc/文件名进行“弱分类”，不得推断真实业务功能。
                                    - 功能性：更偏向 API/命令行工具/解析能力/标准实现等“对外能力相关”的描述（必须能从 desc 直接看出来）。
                                    - 非功能性：测试、构建、兼容性、错误处理、线程安全、性能、可维护性、重构、工具链、基础设施等（同样必须能从 desc 或文件名看出来）。
                                    - 若无法判断，则把条目放入 non_functional_changes说明。
                                """

def generate_change_summary_structured(
    repo: str,
    version_a: str,
    version_b: str,
    change_type: str,
    module_name: str,
    file_count: int,
    semantics_code: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key_env: str = "DEEPSEEK_API_KEY",
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """
    Stage-1: 对单条 change 生成结构化摘要（JSON dict）。

    输入只包含：
    - repo / versions
    - change_type / module_name / file_count
    - semantics.code (added_files/removed_files: path+desc)
    """
    client = _get_client(api_key_env=api_key_env)

    # 仅提供必要字段，避免 token 浪费
    payload = {
        "repo": repo,
        "version_a": version_a,
        "version_b": version_b,
        "change": {
            "type": change_type,
            "module_name": module_name,
            "file_count": file_count,
            "semantics_code": semantics_code or {"added_files": [], "removed_files": []},
        },
    }

    user_prompt = (
        "请对下面这一条架构变更事件生成“结构化摘要”。\n"
        "注意：只基于该事件字段，不得引入任何外部知识或推断。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CHANGE_SUMMARY_SYSTEM_PROMPT_ZH},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = resp.choices[0].message.content or ""
    # 强制 JSON 解析：若模型输出夹杂文本，这里会抛异常，方便你发现 prompt/模型问题
    return text



def main() -> None:
    """
    右键运行自测：
    - 需要你已设置环境变量 DEEPSEEK_API_KEY
    - 会用一个极小的 IR 示例生成 markdown
    """
    demo_ir = {
        "meta": {"version_a": "vA", "version_b": "vB"},
        "quality": {
            "stable_file_universe": True,
            "module_count_delta_large": True,
            "notes": ["Module count changes significantly; interpret with caution."],
        },
        "entities": {"files": {"count_a": 10, "count_b": 10, "added": [], "removed": []}},
        "changes": [
            {
                "id": "CHG-0001",
                "type": "file_reassigned",
                "confidence": 0.9,
                "summary": "File src/a.c reassigned from module X#1 to Y#1.",
                "detail": {"file": "src/a.c", "from_module_uid": "X#1", "to_module_uid": "Y#1"},
                "evidence": [{"kind": "NamedClusters", "ref": "file:src/a.c", "note": "ownership differs"}],
            },
            {
                "id": "CHG-0002",
                "type": "quality_warning",
                "confidence": 1.0,
                "summary": "Module count changes significantly between versions; interpret with caution.",
                "detail": {"flag": "module_count_delta_large"},
                "evidence": [{"kind": "Derived", "ref": "quality:module_count_delta_large", "note": ""}],
            },
        ],
    }

    md = generate_markdown_from_ir(demo_ir)
    print(md)


if __name__ == "__main__":
    main()
