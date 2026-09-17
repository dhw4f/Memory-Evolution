"""LLM 客户端 — Mock + Real 双模式。

MockLLM 按 prompt 关键词匹配任务（abstract / merge / decide_conflict），
输出确定性的 JSON 字符串，保证测试稳定。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# LLMClient — 抽象接口
# ---------------------------------------------------------------------------
class LLMClient:
    """LLM 客户端统一接口。"""

    def __init__(self, mode: str = "auto", model: str = "gpt-4o-mini") -> None:
        self.mode = mode  # auto / mock / real
        self.model = model

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock" or (
            self.mode == "auto" and not os.getenv("OPENAI_API_KEY")
        )

    def invoke(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError

    def invoke_json(self, prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """调用 LLM 并尝试解析为 JSON。失败时返回 {"_error": ...}。"""
        raw = self.invoke(prompt)
        return _try_parse_json(raw)


# ---------------------------------------------------------------------------
# MockLLM — 模板化、确定性返回
# ---------------------------------------------------------------------------
class MockLLM(LLMClient):
    """基于 prompt 关键词匹配的 Mock LLM。

    关键能力：
    - abstract: 从对话中提取关键事实
    - merge: 合并两条记忆
    - decide_conflict: 决定保留哪条
    - respond: 普通对话回应（demo 用）
    """

    _STOPWORDS = {
        "的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
        "和", "与", "或", "但", "也", "都", "很", "就", "要", "有",
        "对", "这", "那", "啊", "吧", "呢", "吗", "嗯", "哦", "哈",
        "，", "。", "、", "！", "？", "\n",
    }

    def __init__(self) -> None:
        super().__init__(mode="mock", model="mock-llm")
        self._cache: dict[str, str] = {}

    def invoke(self, prompt: str, system: str = "") -> str:
        # 缓存：相同 prompt → 相同结果
        cache_key = hashlib.md5(f"{system}|{prompt}".encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        text = prompt + "\n" + system
        text_lower = text.lower()

        # 任务识别
        if "提取关键事实" in text or "abstract" in text_lower or '"facts"' in text:
            result = self._mock_abstract(prompt)
        elif "合并" in text and ("记忆" in text or "merge" in text_lower):
            result = self._mock_merge(prompt)
        elif "决策" in text or "decide" in text_lower or "冲突" in text:
            result = self._mock_decide_conflict(prompt)
        else:
            result = self._mock_respond(prompt)

        self._cache[cache_key] = result
        return result

    # ---- abstract: 提取关键事实 ----
    def _extract_keyphrases(self, text: str) -> list[str]:
        """从文本中抽取关键短语（带上下文）作为事实。"""
        # 清理标点为分隔符
        cleaned = re.sub(r"[，。！？、；：,.!?;:()（）\[\]【】\"\']+", " ", text)
        # 优先 jieba posseg 提取名词短语
        phrases: list[str] = []
        try:
            import jieba.posseg as pseg  # type: ignore

            tokens_with_pos = [
                (w.word, w.flag) for w in pseg.cut(cleaned)
                if w.word.strip() and w.word not in self._STOPWORDS and len(w.word) >= 2
            ]
            # 拼接连续名词为短语
            current: list[str] = []
            for tok, pos in tokens_with_pos:
                if pos.startswith("n") or pos.startswith("vn"):
                    current.append(tok)
                else:
                    if current:
                        phrases.append("".join(current))
                        current = []
                if len(phrases) >= 6:
                    break
            if current:
                phrases.append("".join(current))
        except ImportError:
            phrases = [
                w for w in cleaned.split()
                if w not in self._STOPWORDS and len(w) >= 2
            ][:6]

        return phrases[:6]

    def _mock_abstract(self, prompt: str) -> str:
        # 抽取上下文（对话原文）
        context_match = re.search(r"对话[:：](.+)", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else prompt

        phrases = self._extract_keyphrases(context)

        if not phrases:
            facts = [{"fact": "用户与助手进行了一次对话", "importance": 0.4}]
        else:
            # 每条短语单独作为一条事实（让 consolidation 自然产生多条记忆）
            facts = []
            seen: set[str] = set()
            for p in phrases:
                if p in seen:
                    continue
                seen.add(p)
                imp = 0.6 + min(0.3, len(p) * 0.04)
                facts.append({
                    "fact": p,  # 直接存短语，让 Embedding 自然区分
                    "importance": round(min(0.95, imp), 2),
                })
        return json.dumps({"facts": facts}, ensure_ascii=False)

    # ---- merge: 合并两条记忆 ----
    def _mock_merge(self, prompt: str) -> str:
        # 提取 A: 和 B: 后的内容
        a_match = re.search(r"A[:：]\s*(.+?)(?:\n|B[:：])", prompt, re.DOTALL)
        b_match = re.search(r"B[:：]\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
        a_text = a_match.group(1).strip() if a_match else ""
        b_text = b_match.group(1).strip() if b_match else ""
        merged = f"{a_text};{b_text}".strip(";")
        if not merged:
            merged = "合并后的记忆内容"
        return json.dumps({"merged": merged, "action": "merge"}, ensure_ascii=False)

    # ---- decide_conflict: 决定保留哪条 ----
    def _mock_decide_conflict(self, prompt: str) -> str:
        # 简单策略：基于"最新胜出"原则
        # 但若明显相同/包含关系，返回 merge
        a_match = re.search(r"A[:：]\s*(.+?)(?:\n|B[:：])", prompt, re.DOTALL)
        b_match = re.search(r"B[:：]\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
        a_text = a_match.group(1).strip() if a_match else ""
        b_text = b_match.group(1).strip() if b_match else ""

        if a_text and b_text:
            # 若一方包含另一方，倾向 merge
            if a_text in b_text or b_text in a_text:
                return json.dumps({"action": "merge", "winner": "new"}, ensure_ascii=False)
            # 若语义相反（否定词），用新覆盖
            if any(neg in b_text for neg in ["不", "没", "无"]) and not any(
                neg in a_text for neg in ["不", "没", "无"]
            ):
                return json.dumps({"action": "update", "winner": "new"}, ensure_ascii=False)

        return json.dumps({"action": "update", "winner": "new"}, ensure_ascii=False)

    # ---- respond: 普通对话回应 ----
    def _mock_respond(self, prompt: str) -> str:
        # 简单模板回应
        return "好的,我记住了。"


# ---------------------------------------------------------------------------
# OpenAILLM — 真实 OpenAI 客户端
# ---------------------------------------------------------------------------
class OpenAILLM(LLMClient):
    """使用 langchain-openai 的真实 LLM。"""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(mode="real", model=model)
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            self._client = ChatOpenAI(model=model, temperature=0)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要安装 langchain-openai") from exc

    def invoke(self, prompt: str, system: str = "") -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        resp = self._client.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _try_parse_json(raw: str) -> dict[str, Any]:
    """从字符串中尝试提取 JSON。失败时返回包含 raw 的错误包。"""
    # 尝试直接解析
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试从 ```json ... ``` 代码块提取
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找第一个 { 到最后一个 } 的内容
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"_error": "json_parse_failed", "_raw": raw}


def make_llm(mode: str = "auto", model: str = "gpt-4o-mini") -> LLMClient:
    """根据 mode 返回对应 LLM 实例。"""
    resolved = mode
    if mode == "auto":
        resolved = "real" if os.getenv("OPENAI_API_KEY") else "mock"

    if resolved == "mock":
        return MockLLM()
    if resolved == "real":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("使用 real 模式需要设置 OPENAI_API_KEY")
        return OpenAILLM(model=model)
    raise ValueError(f"未知 llm mode: {mode}")
