"""统一文件存储层 — 管理 JSON 与 Markdown 的读写。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# MemoryRecord — 统一记忆记录 schema
# ---------------------------------------------------------------------------
class MemoryRecord(BaseModel):
    """所有层（episodic/semantic/procedural）共用的记录结构。"""

    id: str
    type: Literal["episodic", "semantic", "procedural"]
    content: str
    embedding: list[float] | None = None
    importance: float = 0.5  # 0-1, 检索权重
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str | None = None
    access_count: int = 0
    source_ids: list[str] = Field(default_factory=list)  # 派生自哪些 episodic
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Storage — 统一文件存储接口
# ---------------------------------------------------------------------------
class Storage:
    """统一管理 ./data/{episodic,semantic,procedural,graph}/ 目录。"""

    def __init__(self, base_dir: str = "./data") -> None:
        self.base_dir = Path(base_dir)
        self.episodic_dir = self.base_dir / "episodic"
        self.semantic_dir = self.base_dir / "semantic"
        self.procedural_dir = self.base_dir / "procedural"
        self.graph_dir = self.base_dir / "graph"
        self._ensure_dirs()

    # ---- 目录管理 ----
    def _ensure_dirs(self) -> None:
        for d in (self.episodic_dir, self.semantic_dir, self.procedural_dir, self.graph_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- 路径辅助 ----
    def episodic_path(self, session_id: str) -> Path:
        return self.episodic_dir / f"{session_id}.md"

    def semantic_path(self, memory_id: str) -> Path:
        return self.semantic_dir / f"{memory_id}.json"

    def procedural_path(self) -> Path:
        return self.procedural_dir / "rules.md"

    def graph_path(self) -> Path:
        return self.graph_dir / "memory_graph.json"

    def meta_path(self) -> Path:
        return self.base_dir / "meta.json"

    # ---- Markdown 读写 ----
    def read_md(self, path: Path | str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def append_md(self, path: Path | str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)

    def write_md(self, path: Path | str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    # ---- JSON 读写 ----
    def read_json(self, path: Path | str) -> Any:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def write_json(self, path: Path | str, data: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- 元数据 ----
    def load_meta(self) -> dict[str, Any]:
        return self.read_json(self.meta_path()) or {}

    def save_meta(self, meta: dict[str, Any]) -> None:
        self.write_json(self.meta_path(), meta)

    def reset(self) -> None:
        """清空所有存储（测试用）。"""
        import shutil

        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self._ensure_dirs()
