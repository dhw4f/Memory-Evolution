"""三层记忆类 — Episodic / Semantic / Procedural。

参考 LangMem 的分层模型：
- Episodic: 原始对话情景（Markdown）
- Semantic: 抽象事实/偏好（JSON + 嵌入）
- Procedural: 行为规则（Markdown）
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EmbeddingClient
from .storage import MemoryRecord, Storage


# ---------------------------------------------------------------------------
# EpisodicMemory — 情景层
# ---------------------------------------------------------------------------
class EpisodicMemory:
    """情景记忆 — 原始对话，按 session_id 组织成 Markdown 文件。"""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._counter = 0  # 本进程内的轮次计数

    def add_turn(self, session_id: str, role: str, content: str) -> str:
        """追加一轮对话。返回 turn_id。"""
        turn_id = f"{session_id}-{self._counter:04d}"
        self._counter += 1
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        block = (
            f"\n## {ts} | {role}\n"
            f"<!-- turn_id: {turn_id} -->\n"
            f"{content}\n"
        )
        self.storage.append_md(self.storage.episodic_path(session_id), block)
        return turn_id

    def get_session(self, session_id: str) -> list[dict[str, str]]:
        """读取一个会话的全部 turn。"""
        text = self.storage.read_md(self.storage.episodic_path(session_id))
        return self._parse_turns(text)

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID。"""
        if not self.storage.episodic_dir.exists():
            return []
        return sorted(p.stem for p in self.storage.episodic_dir.glob("*.md"))

    def recent(self, n: int = 10) -> list[dict[str, str]]:
        """从所有会话中取最近 n 轮。"""
        all_turns: list[dict[str, str]] = []
        for sid in self.list_sessions():
            all_turns.extend(self.get_session(sid))
        # 按时间戳倒序
        all_turns.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
        return all_turns[:n]

    def count(self) -> int:
        """统计总轮次。"""
        total = 0
        for sid in self.list_sessions():
            total += len(self.get_session(sid))
        return total

    @staticmethod
    def _parse_turns(text: str) -> list[dict[str, str]]:
        """解析 Markdown 中的 turns。"""
        if not text:
            return []
        pattern = re.compile(
            r"##\s+(\S+)\s*\|\s*(\w+)\s*\n<!--\s*turn_id:\s*(\S+)\s*-->\n(.*?)(?=\n##\s|\Z)",
            re.DOTALL,
        )
        turns: list[dict[str, str]] = []
        for m in pattern.finditer(text):
            ts, role, turn_id, content = m.groups()
            turns.append({
                "timestamp": ts,
                "role": role.strip(),
                "turn_id": turn_id.strip(),
                "content": content.strip(),
            })
        return turns


# ---------------------------------------------------------------------------
# SemanticMemory — 语义层
# ---------------------------------------------------------------------------
class SemanticMemory:
    """语义记忆 — 抽象事实/偏好，支持语义检索与时间衰减。"""

    def __init__(
        self,
        storage: Storage,
        embedding: EmbeddingClient,
        decay_lambda: float = 0.01,
    ) -> None:
        self.storage = storage
        self.embedding = embedding
        self.decay_lambda = decay_lambda

    # ---- CRUD ----
    def add(
        self,
        content: str,
        importance: float = 0.5,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """新增一条 semantic 记忆。"""
        mem_id = f"sem_{uuid.uuid4().hex[:8]}"
        vec = self.embedding.embed(content)
        record = MemoryRecord(
            id=mem_id,
            type="semantic",
            content=content,
            embedding=vec.tolist(),
            importance=max(0.0, min(1.0, importance)),
            source_ids=source_ids or [],
            metadata=metadata or {},
        )
        self._save(record)
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        data = self.storage.read_json(self.storage.semantic_path(memory_id))
        if data is None:
            return None
        return MemoryRecord.from_dict(data)

    def update(
        self,
        memory_id: str,
        new_content: str,
        importance: float | None = None,
        strategy: str = "recency_wins",
    ) -> MemoryRecord | None:
        """更新记忆内容。

        strategy:
          - recency_wins: 直接覆盖
          - merge: 与原内容拼接去重
        """
        existing = self.get(memory_id)
        if existing is None:
            return None

        if strategy == "merge":
            seen = set()
            merged_tokens: list[str] = []
            for piece in [existing.content, new_content]:
                for token in re.split(r"[，。；、,.;\s]+", piece):
                    token = token.strip()
                    if token and token not in seen:
                        seen.add(token)
                        merged_tokens.append(token)
            new_content = "，".join(merged_tokens) if merged_tokens else new_content

        vec = self.embedding.embed(new_content)
        existing.content = new_content
        existing.embedding = vec.tolist()
        if importance is not None:
            existing.importance = max(0.0, min(1.0, importance))
        existing.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(existing)
        return existing

    def delete(self, memory_id: str) -> bool:
        path = self.storage.semantic_path(memory_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def all(self) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        if not self.storage.semantic_dir.exists():
            return records
        for p in self.storage.semantic_dir.glob("*.json"):
            data = self.storage.read_json(p)
            if data:
                records.append(MemoryRecord.from_dict(data))
        return records

    def count(self) -> int:
        if not self.storage.semantic_dir.exists():
            return 0
        return len(list(self.storage.semantic_dir.glob("*.json")))

    # ---- 检索 ----
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """语义检索：score = importance × cosine_sim × exp(-λ × elapsed_days)。"""
        query_vec = self.embedding.embed(query)
        now = datetime.now(timezone.utc).timestamp()
        scored: list[tuple[MemoryRecord, float]] = []

        for record in self.all():
            if record.embedding is None:
                continue
            mem_vec = np.asarray(record.embedding)
            sim = self.embedding.cosine_sim(query_vec, mem_vec)

            # 时间衰减
            elapsed_days = 0.0
            try:
                ts = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
                elapsed_days = max(0.0, (now - ts.timestamp()) / 86400)
            except (ValueError, AttributeError):
                pass
            decay = math.exp(-self.decay_lambda * elapsed_days)

            score = record.importance * max(0.0, sim) * decay
            if score >= min_score:
                scored.append((record, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 命中后更新访问计数
        for record, score in scored[:top_k]:
            self._bump_access(record)

        return scored[:top_k]

    def find_similar(
        self,
        content: str,
        threshold: float = 0.82,
    ) -> list[tuple[MemoryRecord, float]]:
        """找相似度 >= threshold 的现有记忆。"""
        vec = self.embedding.embed(content)
        matched: list[tuple[MemoryRecord, float]] = []
        for record in self.all():
            if record.embedding is None:
                continue
            sim = self.embedding.cosine_sim(vec, np.asarray(record.embedding))
            if sim >= threshold:
                matched.append((record, sim))
        matched.sort(key=lambda x: x[1], reverse=True)
        return matched

    # ---- 内部 ----
    def _save(self, record: MemoryRecord) -> None:
        self.storage.write_json(
            self.storage.semantic_path(record.id),
            record.to_dict(),
        )

    def _bump_access(self, record: MemoryRecord) -> None:
        record.access_count += 1
        record.last_accessed = datetime.now(timezone.utc).isoformat()
        self._save(record)


# ---------------------------------------------------------------------------
# ProceduralMemory — 程序层
# ---------------------------------------------------------------------------
class ProceduralMemory:
    """程序记忆 — 行为规则 / 系统指令，Markdown 存储。"""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.path = storage.procedural_path()

    def add_rule(self, rule: str, category: str = "general") -> None:
        """追加一条规则。"""
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        block = f"\n### [{category}] {ts}\n- {rule}\n"
        self.storage.append_md(self.path, block)

    def list_rules(self, category: str | None = None) -> list[str]:
        """列出所有规则，可按 category 过滤。"""
        text = self.storage.read_md(self.path)
        if not text:
            return []
        rules: list[str] = []
        pattern = re.compile(
            r"###\s*\[(\w+)\]\s*\S+\n-\s*(.+?)(?=\n###|\Z)", re.DOTALL
        )
        for m in pattern.finditer(text):
            cat, rule = m.groups()
            if category is None or cat == category:
                rules.append(rule.strip())
        return rules

    def remove_rule(self, rule_substring: str) -> int:
        """删除包含子串的规则，返回删除条数。"""
        text = self.storage.read_md(self.path)
        if not text:
            return 0
        new_lines: list[str] = []
        removed = 0
        for line in text.splitlines(keepends=True):
            if line.strip().startswith("- ") and rule_substring in line:
                removed += 1
                continue
            new_lines.append(line)
        if removed > 0:
            self.storage.write_md(self.path, "".join(new_lines))
        return removed

    def format_for_prompt(self) -> str:
        """格式化为可注入 prompt 的字符串。"""
        rules = self.list_rules()
        if not rules:
            return "(无行为规则)"
        return "\n".join(f"- {r}" for r in rules)

    def count(self) -> int:
        return len(self.list_rules())
