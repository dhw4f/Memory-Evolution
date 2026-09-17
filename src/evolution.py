"""MemoryEvolutionOrchestrator — 协调三层记忆 + 进化机制。

对外暴露统一接口：
  - on_message(): 接收对话 turn，触发存储和可能的进化
  - retrieve_context(): 检索相关记忆，格式化为 prompt 片段
  - status(): 系统状态快照
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .consolidation import ConsolidationEngine, ConsolidationResult
from .embeddings import EmbeddingClient, make_embedding
from .graph import MemoryGraph
from .llm import LLMClient, make_llm
from .memory_layers import EpisodicMemory, ProceduralMemory, SemanticMemory
from .storage import MemoryRecord, Storage


@dataclass
class EvolutionConfig:
    """演化器配置。"""

    base_dir: str = "./data"
    llm_mode: str = "auto"
    embedding_mode: str = "auto"
    embedding_dim: int = 256
    llm_model: str = "gpt-4o-mini"
    consolidation_threshold: int = 5
    similarity_threshold: float = 0.82
    graph_edge_threshold: float = 0.6
    decay_lambda: float = 0.01
    retrieval_top_k: int = 5
    graph_neighbor_depth: int = 1


class MemoryEvolutionOrchestrator:
    """记忆进化编排器 — 把三层记忆 + LLM + 嵌入 + 图 + consolidation 串起来。"""

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()

        # 基础设施
        self.storage = Storage(self.config.base_dir)
        self.llm: LLMClient = make_llm(
            mode=self.config.llm_mode, model=self.config.llm_model
        )
        self.embedding: EmbeddingClient = make_embedding(
            mode=self.config.embedding_mode, dim=self.config.embedding_dim
        )

        # 三层记忆
        self.episodic = EpisodicMemory(self.storage)
        self.semantic = SemanticMemory(
            self.storage, self.embedding, decay_lambda=self.config.decay_lambda
        )
        self.procedural = ProceduralMemory(self.storage)

        # 关联图
        self.graph = MemoryGraph(
            self.storage,
            self.embedding,
            edge_threshold=self.config.graph_edge_threshold,
        )
        self.graph.load()

        # 进化引擎
        self.consolidator = ConsolidationEngine(
            llm=self.llm,
            embedding=self.embedding,
            episodic=self.episodic,
            semantic=self.semantic,
            consolidation_threshold=self.config.consolidation_threshold,
            similarity_threshold=self.config.similarity_threshold,
        )

    # ---- 对话入口 ----
    def on_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        """接收一条消息。返回是否触发进化 + 进化结果摘要。"""
        turn_id = self.episodic.add_turn(session_id, role, content)
        triggered = self.consolidator.observe_turn()

        result: dict[str, Any] = {
            "turn_id": turn_id,
            "triggered": triggered,
            "result": None,
        }

        if triggered:
            cons = self.consolidator.last_result
            self._post_consolidate(cons)
            result["result"] = cons.summary() if cons else None

        return result

    def _post_consolidate(self, cons: ConsolidationResult | None) -> None:
        """consolidation 完成后重建关联图。"""
        if cons is None:
            return
        if cons.added or cons.updated or cons.deleted:
            self.graph.rebuild(self.semantic)

    # ---- 检索 ----
    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
    ) -> str:
        """检索相关记忆,拼接为 prompt 片段。

        流程:
          1. semantic.search(query) → 候选
          2. 对 top-1 扩展 graph.neighbors()
          3. 追加 procedural rules
          4. 拼接返回
        """
        k = top_k or self.config.retrieval_top_k

        # 1. semantic 检索
        hits = self.semantic.search(query, top_k=k)
        lines: list[str] = ["[语义记忆]"]
        primary_ids: list[str] = []
        for rec, score in hits:
            lines.append(f"- ({score:.2f}) {rec.content}")
            primary_ids.append(rec.id)

        # 2. 关联扩展（取相似记忆）
        expanded_ids: set[str] = set()
        for pid in primary_ids[:2]:  # 只对前 2 个扩展
            for nb_id, weight in self.graph.neighbors_with_weight(pid, depth=self.config.graph_neighbor_depth)[:3]:
                if nb_id not in primary_ids and weight > 0.3:
                    expanded_ids.add(nb_id)

        if expanded_ids:
            lines.append("\n[关联记忆]")
            for nb_id in list(expanded_ids)[:3]:
                rec = self.semantic.get(nb_id)
                if rec:
                    lines.append(f"- {rec.content}")

        # 3. procedural rules
        rules_text = self.procedural.format_for_prompt()
        lines.append(f"\n[行为规则]\n{rules_text}")

        return "\n".join(lines)

    # ---- 状态 ----
    def status(self) -> dict[str, Any]:
        """返回系统状态快照。"""
        return {
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "procedural_count": self.procedural.count(),
            "graph": self.graph.stats(),
            "consolidator": self.consolidator.status(),
            "llm_mode": "mock" if self.llm.is_mock else "real",
            "embedding_mode": "mock" if self.embedding.is_mock else "real",
        }

    # ---- 维护 ----
    def reset(self) -> None:
        """清空所有数据(测试用)。"""
        self.storage.reset()
        # 重置 in-memory state
        self.graph.g.clear()
        self.consolidator._since_last = 0
        self.consolidator.last_result = None
        self.consolidator.total_consolidations = 0

    def export_graph_html(self, path: str = "./data/graph.html") -> bool:
        """导出关联图为 HTML。"""
        if self.graph.g.number_of_nodes() == 0:
            self.graph.rebuild(self.semantic)
        return self.graph.export_html(path)
