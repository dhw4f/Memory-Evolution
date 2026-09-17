"""Consolidation Engine — 记忆进化的核心机制。

五阶段流水线（参考 Mem0 / LangMem）：
  Extract → Integrate → Store → Retrieve → Forget

在本模块中实现为：
  1. 收集最近 episodic（Extract）
  2. Embedding 相似度初筛（Union-Find 聚类）
  3. LLM 抽象 + 冲突处理（Integrate）
  4. 写入 semantic（Store）
  5. 返回结果统计
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .embeddings import EmbeddingClient
from .llm import LLMClient
from .memory_layers import EpisodicMemory, SemanticMemory


class Action(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


@dataclass
class ConsolidationResult:
    """一次 consolidation 的结果统计。"""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    noop: int = 0
    clusters_processed: int = 0
    facts_extracted: int = 0
    detail: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "added": len(self.added),
            "updated": len(self.updated),
            "deleted": len(self.deleted),
            "noop": self.noop,
            "clusters": self.clusters_processed,
            "facts": self.facts_extracted,
        }


# ---------------------------------------------------------------------------
# Union-Find — 简易并查集用于聚类
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# ConsolidationEngine — 进化主引擎
# ---------------------------------------------------------------------------
class ConsolidationEngine:
    """记忆进化核心引擎。

    职责：
      - 监听 episodic 累计，达到阈值触发 consolidate()
      - 把 episodic 聚类、抽象为 semantic
      - 处理冲突（recency_wins + LLM 仲裁）
      - 返回操作结果
    """

    def __init__(
        self,
        llm: LLMClient,
        embedding: EmbeddingClient,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        consolidation_threshold: int = 5,
        similarity_threshold: float = 0.82,
        recent_window: int = 10,
    ) -> None:
        self.llm = llm
        self.embedding = embedding
        self.episodic = episodic
        self.semantic = semantic
        self.consolidation_threshold = consolidation_threshold
        self.similarity_threshold = similarity_threshold
        self.recent_window = recent_window

        self._since_last = 0  # 距上次 consolidation 的新 turn 数
        self.last_result: ConsolidationResult | None = None
        self.total_consolidations = 0

    # ---- 触发判定 ----
    def observe_turn(self) -> bool:
        """通知引擎：有一条新 turn。返回是否触发了 consolidation。"""
        self._since_last += 1
        if self._since_last >= self.consolidation_threshold:
            self.consolidate()
            return True
        return False

    def force(self) -> ConsolidationResult:
        """强制触发一次 consolidation（demo / 测试用）。"""
        return self.consolidate()

    # ---- 主入口 ----
    def consolidate(self) -> ConsolidationResult:
        """执行一次完整的 consolidation。"""
        result = ConsolidationResult()

        # 1. Extract: 收集最近 episodic
        recent_turns = self.episodic.recent(self.recent_window)
        if not recent_turns:
            self._since_last = 0
            self.last_result = result
            return result

        # 2. 嵌入 + 聚类
        clusters = self._cluster_turns(recent_turns)
        result.clusters_processed = len(clusters)

        # 3-4. 对每簇抽象 + Integrate
        for cluster in clusters:
            facts = self._abstract_cluster(cluster)
            result.facts_extracted += len(facts)

            for fact_info in facts:
                action, mem_id = self._integrate_fact(fact_info, cluster)
                if action == Action.ADD:
                    result.added.append(mem_id or "")
                elif action == Action.UPDATE:
                    result.updated.append(mem_id or "")
                elif action == Action.DELETE:
                    result.deleted.append(mem_id or "")
                else:
                    result.noop += 1
                result.detail.append(
                    {
                        "action": action.value,
                        "memory_id": mem_id,
                        "fact": fact_info.get("fact"),
                        "importance": fact_info.get("importance", 0.5),
                    }
                )

        self._since_last = 0
        self.last_result = result
        self.total_consolidations += 1
        return result

    # ---- 聚类 ----
    def _cluster_turns(self, turns: list[dict[str, str]]) -> list[list[dict[str, str]]]:
        """用 Embedding 相似度对 turn 聚类。"""
        if not turns:
            return []

        # 计算所有 turn 的嵌入
        texts = [t["content"] for t in turns]
        vectors = self.embedding.embed_batch(texts)

        uf = _UnionFind(len(turns))
        for i in range(len(turns)):
            for j in range(i + 1, len(turns)):
                sim = self.embedding.cosine_sim(vectors[i], vectors[j])
                if sim >= self.similarity_threshold:
                    uf.union(i, j)

        # 按根分组
        groups: dict[int, list[int]] = {}
        for i in range(len(turns)):
            r = uf.find(i)
            groups.setdefault(r, []).append(i)

        return [[turns[i] for i in idxs] for idxs in groups.values()]

    # ---- 抽象 ----
    def _abstract_cluster(self, cluster: list[dict[str, str]]) -> list[dict[str, Any]]:
        """把一簇 turn 抽象为候选事实列表。"""
        context = "\n".join(f"[{t['role']}] {t['content']}" for t in cluster)

        prompt = (
            "请从以下对话中提取关键事实,用于长期记忆。\n"
            "输出严格的 JSON: {\"facts\": [{\"fact\": \"...\", \"importance\": 0.0-1.0}]}\n\n"
            f"对话:\n{context}"
        )

        result = self.llm.invoke_json(prompt)
        if "_error" in result:
            # Mock 失败兜底:整段作为一条事实
            return [{"fact": context[:200], "importance": 0.5}]

        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return []
        # 校验每条 fact
        clean: list[dict[str, Any]] = []
        for item in facts:
            if isinstance(item, dict) and "fact" in item:
                importance = item.get("importance", 0.5)
                try:
                    importance = float(importance)
                except (TypeError, ValueError):
                    importance = 0.5
                clean.append({
                    "fact": str(item["fact"]),
                    "importance": max(0.0, min(1.0, importance)),
                })
        return clean

    # ---- Integrate ----
    def _integrate_fact(
        self,
        fact_info: dict[str, Any],
        cluster: list[dict[str, str]],
    ) -> tuple[Action, str | None]:
        """对一条候选事实执行 ADD / UPDATE / DELETE / NOOP。"""
        fact = fact_info["fact"]
        importance = fact_info["importance"]
        source_ids = [t["turn_id"] for t in cluster if t.get("turn_id")]

        # 查重
        existing = self.semantic.find_similar(fact, threshold=self.similarity_threshold)

        if not existing:
            # ADD
            rec = self.semantic.add(
                content=fact,
                importance=importance,
                source_ids=source_ids,
                metadata={"derived_from_cluster_size": len(cluster)},
            )
            return Action.ADD, rec.id

        # 冲突处理
        existing_record, sim = existing[0]
        decision = self._resolve_conflict(existing_record.content, fact)

        if decision == "noop":
            return Action.NOOP, existing_record.id
        if decision == "merge":
            self.semantic.update(
                existing_record.id,
                fact,
                importance=max(importance, existing_record.importance),
                strategy="merge",
            )
            return Action.UPDATE, existing_record.id
        # decision == "update"（默认 recency-wins）
        self.semantic.update(
            existing_record.id,
            fact,
            importance=max(importance, existing_record.importance),
        )
        return Action.UPDATE, existing_record.id

    def _resolve_conflict(self, existing: str, new: str) -> str:
        """让 LLM 决定如何处理冲突。返回 update / merge / noop。"""
        prompt = (
            "判断两条记忆的关系并决策。\n"
            "A: " + existing + "\n"
            "B: " + new + "\n"
            "请输出 JSON: {\"action\": \"update|merge|noop\", \"winner\": \"new|old\"}\n"
            "- update: 新记忆覆盖旧记忆\n"
            "- merge: 合并两条记忆\n"
            "- noop: 内容已包含,无需操作"
        )
        result = self.llm.invoke_json(prompt)
        action = result.get("action", "update")
        if action not in {"update", "merge", "noop"}:
            action = "update"
        return action

    # ---- 状态 ----
    def status(self) -> dict[str, Any]:
        return {
            "since_last": self._since_last,
            "threshold": self.consolidation_threshold,
            "total_consolidations": self.total_consolidations,
            "last_result": self.last_result.summary() if self.last_result else None,
        }
