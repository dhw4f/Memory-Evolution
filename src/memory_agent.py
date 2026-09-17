"""EvolvingMemoryAgent — 面向用户的顶层接口。

封装:
  - retrieve_context (自动注入相关记忆)
  - LLM 调用
  - episodic 写入
  - consolidation 触发
"""

from __future__ import annotations

from typing import Any

from .evolution import EvolutionConfig, MemoryEvolutionOrchestrator


class EvolvingMemoryAgent:
    """面向用户的统一智能体入口。"""

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()
        self.orchestrator = MemoryEvolutionOrchestrator(self.config)
        self.session_id: str = "default"

    @property
    def episodic(self):
        return self.orchestrator.episodic

    @property
    def semantic(self):
        return self.orchestrator.semantic

    @property
    def procedural(self):
        return self.orchestrator.procedural

    @property
    def graph(self):
        return self.orchestrator.graph

    # ---- 对话 ----
    def chat(self, user_message: str, session_id: str | None = None) -> str:
        """单轮对话:检索上下文 → LLM 回应 → 写入记忆。"""
        sid = session_id or self.session_id
        self.session_id = sid

        # 1. 检索上下文
        context = self.orchestrator.retrieve_context(user_message)

        # 2. LLM 回应
        response = self.orchestrator.llm.invoke(
            prompt=user_message,
            system=(
                "你是一个有长期记忆的助手。你会参考以下记忆上下文来回应用户。\n"
                f"上下文:\n{context}"
            ),
        )

        # 3. 写入 episodic（user + assistant）
        user_result = self.orchestrator.on_message(sid, "user", user_message)
        asst_result = self.orchestrator.on_message(sid, "assistant", response)

        return response

    def observe(self, role: str, content: str, session_id: str | None = None) -> dict[str, Any]:
        """只观察不回应(用于批量导入历史)。"""
        sid = session_id or self.session_id
        return self.orchestrator.on_message(sid, role, content)

    # ---- 程序记忆 ----
    def add_rule(self, rule: str, category: str = "general") -> None:
        self.procedural.add_rule(rule, category)

    def list_rules(self, category: str | None = None) -> list[str]:
        return self.procedural.list_rules(category)

    # ---- 检索 ----
    def retrieve(self, query: str, top_k: int | None = None) -> str:
        return self.orchestrator.retrieve_context(query, top_k=top_k)

    # ---- 维护 ----
    def status(self) -> dict[str, Any]:
        return self.orchestrator.status()

    def reset(self) -> None:
        self.orchestrator.reset()

    def force_consolidate(self) -> dict[str, Any]:
        """手动触发一次 consolidation。"""
        result = self.orchestrator.consolidator.force()
        self.orchestrator._post_consolidate(result)
        return result.summary()

    def export_graph_html(self, path: str = "./data/graph.html") -> bool:
        return self.orchestrator.export_graph_html(path)
