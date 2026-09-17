"""MemoryGraph — 基于 NetworkX 的记忆关联网络。

节点 = semantic 记忆
边类型：
  - similar: cosine sim > edge_threshold
  - derived_from: 来自同一簇 episodic
  - conflicts: 冲突但保留（当前未启用）
"""

from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
import numpy as np

from .embeddings import EmbeddingClient
from .memory_layers import SemanticMemory
from .storage import Storage


class MemoryGraph:
    """记忆关联图。"""

    def __init__(
        self,
        storage: Storage,
        embedding: EmbeddingClient,
        edge_threshold: float = 0.6,
    ) -> None:
        self.storage = storage
        self.embedding = embedding
        self.edge_threshold = edge_threshold
        self.g: nx.Graph = nx.Graph()
        self._loaded = False

    # ---- 加载 / 保存 ----
    def load(self) -> None:
        """从磁盘加载图（与 save 格式对齐）。"""
        data = self.storage.read_json(self.storage.graph_path())
        if not data:
            self._loaded = True
            return
        self.g = nx.Graph()
        # 节点可能是 [{id, ...attrs}] 或 [[id, attrs]] 两种格式
        for node in data.get("nodes", []):
            if isinstance(node, dict):
                nid = node.get("id", node.get("name"))
                attrs = {k: v for k, v in node.items() if k not in ("id", "name")}
                if nid is not None:
                    self.g.add_node(nid, **attrs)
        # 边可能是 [[u, v, attrs]] 或 [{source, target, ...}]
        for edge in data.get("edges", []):
            if isinstance(edge, dict):
                u = edge.get("source")
                v = edge.get("target")
                attrs = {
                    k: val
                    for k, val in edge.items()
                    if k not in ("source", "target")
                }
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                u, v = edge[0], edge[1]
                attrs = edge[2] if len(edge) >= 3 and isinstance(edge[2], dict) else {}
            else:
                continue
            if u is not None and v is not None:
                self.g.add_edge(u, v, **attrs)
        self._loaded = True

    def save(self) -> None:
        """序列化到磁盘。"""
        # 使用通用 JSON 格式（避免 networkx 版本差异）
        data = {
            "nodes": [
                {"id": n, **self.g.nodes[n]} for n in self.g.nodes
            ],
            "edges": [
                [u, v, dict(self.g.edges[u, v])] for u, v in self.g.edges
            ],
        }
        self.storage.write_json(self.storage.graph_path(), data)

    # ---- 构建 ----
    def rebuild(self, semantic: SemanticMemory) -> None:
        """根据 semantic 全部记忆重建图。"""
        self.g.clear()
        records = semantic.all()
        if not records:
            return

        # 添加节点
        for r in records:
            self.g.add_node(
                r.id,
                id=r.id,
                type=r.type,
                content_preview=r.content[:80],
                importance=r.importance,
                created_at=r.created_at,
            )

        # 两两计算相似度，加边
        vecs = {r.id: np.asarray(r.embedding) for r in records if r.embedding}
        ids = list(vecs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = self.embedding.cosine_sim(vecs[ids[i]], vecs[ids[j]])
                if sim >= self.edge_threshold:
                    self.g.add_edge(
                        ids[i],
                        ids[j],
                        weight=round(float(sim), 4),
                        relation="similar",
                    )
        self.save()

    def add_node(self, memory_id: str, type_: str = "semantic", **attrs: Any) -> None:
        attrs.setdefault("id", memory_id)
        attrs.setdefault("type", type_)
        self.g.add_node(memory_id, **attrs)

    def add_edge(
        self,
        id_a: str,
        id_b: str,
        weight: float,
        relation: str = "similar",
    ) -> None:
        if self.g.has_edge(id_a, id_b):
            # 取更高权重
            existing = self.g[id_a][id_b].get("weight", 0.0)
            if weight > existing:
                self.g[id_a][id_b]["weight"] = weight
                self.g[id_a][id_b]["relation"] = relation
        else:
            self.g.add_edge(id_a, id_b, weight=weight, relation=relation)

    # ---- 查询 ----
    def neighbors(self, memory_id: str, depth: int = 2) -> list[str]:
        """返回 BFS depth 层内的邻居 ID。"""
        if memory_id not in self.g:
            return []
        visited: set[str] = {memory_id}
        frontier: set[str] = {memory_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for nb in self.g.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(memory_id)
        return list(visited)

    def neighbors_with_weight(
        self, memory_id: str, depth: int = 2
    ) -> list[tuple[str, float]]:
        """返回邻居及其到 memory_id 的累计权重。"""
        if memory_id not in self.g:
            return []

        result: list[tuple[str, float]] = []
        visited: set[str] = {memory_id}
        frontier: list[tuple[str, float]] = [(memory_id, 1.0)]
        for _ in range(depth):
            next_frontier: list[tuple[str, float]] = []
            for node, acc_w in frontier:
                for nb in self.g.neighbors(node):
                    if nb in visited:
                        continue
                    w = self.g[node][nb].get("weight", 0.5)
                    result.append((nb, acc_w * w))
                    visited.add(nb)
                    next_frontier.append((nb, acc_w * w))
            frontier = next_frontier
            if not frontier:
                break
        # 按权重排序去重
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def find_path(self, id_a: str, id_b: str) -> list[str]:
        """两节点间最短路径。"""
        try:
            return list(nx.shortest_path(self.g, id_a, id_b))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def communities(self) -> list[list[str]]:
        """社区检测（主题聚类）。"""
        if self.g.number_of_nodes() == 0:
            return []
        try:
            from networkx.algorithms.community import greedy_modularity_communities

            comms = list(greedy_modularity_communities(self.g))
            return [list(c) for c in comms]
        except Exception:
            # 退化:用连通分量
            return [list(c) for c in nx.connected_components(self.g)]

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "density": round(nx.density(self.g), 4),
            "components": nx.number_connected_components(self.g),
        }

    # ---- 可视化（可选） ----
    def export_html(self, path: str) -> bool:
        """用 pyvis 导出交互式 HTML。"""
        try:
            from pyvis.network import Network  # type: ignore

            net = Network(height="600px", width="100%", notebook=False)
            for n, attrs in self.g.nodes(data=True):
                label = attrs.get("content_preview", n)[:30]
                net.add_node(n, label=label, title=attrs.get("content_preview", ""))
            for u, v, attrs in self.g.edges(data=True):
                net.add_edge(u, v, value=attrs.get("weight", 0.5))
            net.show(path)
            return True
        except ImportError:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.g.nodes),
            "edges": [
                {"source": u, "target": v, **self.g.edges[u, v]}
                for u, v in self.g.edges
            ],
        }

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self.g

    def __iter__(self) -> Iterable[str]:
        return iter(self.g.nodes)
