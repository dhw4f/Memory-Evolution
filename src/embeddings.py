"""嵌入模型 — Mock + Real 双模式。

MockEmbedding 用 jieba 分词 + 多重 hash + 字符 n-gram 构造 dim 维向量：
- 同一文本 → 同一向量（确定性）
- 同主题词 → 高 cosine 相似度
- 跨主题文本 → 低相似度
"""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np

try:
    import jieba  # type: ignore

    _JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JIEBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# EmbeddingClient — 抽象接口
# ---------------------------------------------------------------------------
class EmbeddingClient:
    """嵌入模型统一接口。"""

    def __init__(self, mode: str = "auto", dim: int = 256) -> None:
        self.mode = mode  # auto / mock / real
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock" or (
            self.mode == "auto" and not os.getenv("OPENAI_API_KEY")
        )

    # ---- 必须实现的接口 ----
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# MockEmbedding — 基于 hash 的本地实现
# ---------------------------------------------------------------------------
class MockEmbedding(EmbeddingClient):
    """确定性 hash-based 嵌入，同主题文本产生高相似度向量。"""

    def __init__(self, dim: int = 256) -> None:
        super().__init__(mode="mock", dim=dim)
        # 多重 hash 种子以增加稀疏度
        self._seed_a = 31
        self._seed_b = 131

    # ---- token 化 ----
    def _tokenize(self, text: str) -> list[str]:
        if _JIEBA_AVAILABLE:
            # jieba 精确模式，对中文友好
            return [t for t in jieba.lcut(text) if t.strip()]
        # 退化方案：字符 bigram
        text = text.strip()
        if not text:
            return []
        return [text[i : i + 2] for i in range(len(text) - 1)] or [text]

    def _ngrams(self, text: str, n: int = 3) -> list[str]:
        """字符 n-gram 兜底，捕获未登录词。"""
        text = text.lower().replace(" ", "")
        if len(text) < n:
            return [text]
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    # ---- 哈希到位 ----
    def _hash_to_index(self, token: str, salt: int) -> int:
        h = hashlib.md5(f"{salt}:{token}".encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def _sign(self, token: str, salt: int) -> int:
        h = hashlib.md5(f"sign:{salt}:{token}".encode("utf-8")).hexdigest()
        return 1 if int(h, 16) % 2 == 0 else -1

    # ---- 主入口 ----
    def _embed_uncached(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)

        # 1) token 级别（jieba 或 bigram）— 命中 3 个不同位置(增强同主题聚集)
        for tok in self._tokenize(text):
            for salt in (self._seed_a, self._seed_a * 7, self._seed_a * 13):
                idx = self._hash_to_index(tok, salt)
                sign = self._sign(tok, salt)
                vec[idx] += sign * 2.0

        # 2) 字符 bigram 兜底(权重较低)
        for gram in self._ngrams(text, n=2):
            idx = self._hash_to_index(gram, self._seed_b)
            sign = self._sign(gram, self._seed_b)
            vec[idx] += sign * 1.0

        # 3) L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0  # 避免零向量
        return vec

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self._embed_uncached(text)
        return self._cache[text].copy()


# ---------------------------------------------------------------------------
# OpenAIEmbedding — 真实 OpenAI 嵌入
# ---------------------------------------------------------------------------
class OpenAIEmbedding(EmbeddingClient):
    """使用 langchain-openai 的真实嵌入。"""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        super().__init__(mode="real", dim=dim)
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore

            self._client = OpenAIEmbeddings(model=model)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要安装 langchain-openai 才能使用真实嵌入") from exc

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self._client.embed_query(text)
            self._cache[text] = np.asarray(vec, dtype=np.float64)
        return self._cache[text].copy()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def make_embedding(mode: str = "auto", dim: int = 256) -> EmbeddingClient:
    """根据 mode 返回对应嵌入实例。

    - auto: 有 OPENAI_API_KEY → real，否则 mock
    - mock: 强制 MockEmbedding
    - real: 强制 OpenAIEmbedding（缺 key 抛错）
    """
    resolved = mode
    if mode == "auto":
        resolved = "real" if os.getenv("OPENAI_API_KEY") else "mock"

    if resolved == "mock":
        return MockEmbedding(dim=dim)
    if resolved == "real":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("使用 real 模式需要设置 OPENAI_API_KEY")
        return OpenAIEmbedding(dim=dim)
    raise ValueError(f"未知 embedding mode: {mode}")
