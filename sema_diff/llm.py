"""
llm.py
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


SYSTEM_PROMPT_STRICT = """You are an architecture-change reporting assistant. Your answers should be in Chinese.

Hard rules (must follow):
1) You MUST NOT invent any facts. Only use information present in the provided diff_ir.json content.
2) Every statement about a change MUST cite at least one Change ID in the form [CHG-XXXX]. If you cannot cite a Change ID, do not mention that statement.
3) Do NOT infer code-level or behavioral impact unless the diff explicitly states it in the change detail/evidence. Prefer neutral wording like "reassigned" / "aligned" / "mapped".
4) If a change has confidence < 0.75, explicitly mark it as "Low confidence" in the bullet.
5) If the quality section indicates instability (e.g., module_count_delta_large, duplicate names, incomplete mapping), include a dedicated "Reliability notes" section summarizing those flags without exaggeration.
6) Output must be valid Markdown. Use concise, technical language.

Required output structure:
- Title line: "Architecture Change Report: <version_a> → <version_b>"
- Section 1: "Overview" (2-4 bullets)
- Section 2: "Detected Changes" (group by type: Files / Modules / Components / Quality)
- Section 3: "Reliability notes" (bullets)
- Section 4: "Appendix: Change Index" (a compact list mapping CHG id -> one-line summary)
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
        "Generate a Markdown architecture change report strictly based on the following diff IR JSON.\n\n"
        "diff_ir.json:\n"
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
