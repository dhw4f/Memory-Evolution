# Memory Evole — 智能体记忆自动进化系统

> 让智能体在长对话中自动维护越来越精炼、可复用的记忆，避免上下文窗口爆炸。

## 核心特性

- 🧠 **三层记忆模型**：情景（Episodic）/ 语义（Semantic）/ 程序（Procedural）
- 🔄 **自动进化**：压缩、抽象、合并、冲突处理、衰减
- 🕸️ **关联网络**：基于 NetworkX 的记忆关联图，支持语义检索扩展
- 💾 **本地存储**：Markdown（人类可读）+ JSON（结构化元数据）
- 🧪 **Mock 优先**：无需 API key 即可端到端运行（也可切换真实 LLM）

## 快速开始

```bash
# 安装依赖
pip install -e .

# 运行端到端演示
python demo.py

# 运行测试
pip install -e ".[dev]"
pytest tests/ -v
```

## 架构

```
对话输入 → Episodic 记录（Markdown）
              ↓ 达到阈值（默认 5 轮）
       Consolidation Engine
              ├─ Embedding 相似度聚类
              ├─ LLM 抽象（mock/real）
              ├─ 冲突处理（Recency-wins + LLM 仲裁）
              └─ 写入 Semantic（JSON）
              ↓
       NetworkX 关联图（自动重建）
              ↓
       检索：score = importance × cosine_sim × exp(-λt)
```

## 文件结构

```
memory-evole/
├── pyproject.toml
├── README.md
├── demo.py                    # 端到端演示
├── src/
│   ├── __init__.py
│   ├── storage.py             # 文件存储
│   ├── llm.py                 # LLM 客户端（mock/real）
│   ├── embeddings.py          # 嵌入模型（mock/real）
│   ├── memory_layers.py       # 三层记忆类
│   ├── consolidation.py       # 压缩/合并/抽象
│   ├── graph.py               # NetworkX 关联图
│   ├── evolution.py           # 编排器
│   └── memory_agent.py        # 顶层 Agent
├── tests/
│   └── test_smoke.py          # 冒烟测试
└── data/                      # 运行时生成
    ├── episodic/<session>.md
    ├── semantic/<id>.json
    ├── procedural/rules.md
    └── graph/memory_graph.json
```

## 切换到真实 LLM

设置环境变量 `OPENAI_API_KEY` 即可自动切换到真实模式：

```bash
export OPENAI_API_KEY=sk-xxx
python demo.py
```

无 key 时默认使用 Mock，演示效果一致。

## 调优参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `consolidation_threshold` | 5 | 触发 consolidation 的 episodic 计数阈值 |
| `similarity_threshold` | 0.82 | 记忆合并的相似度阈值 |
| `graph_edge_threshold` | 0.6 | 关联图的边权重阈值 |
| `decay_lambda` | 0.01 | 衰减系数（30 天衰减 ~74%） |
| `embedding_dim` | 256 | 嵌入向量维度 |

## 演示输出示例

```
=== 智能体记忆进化系统演示 ===

--- 会话 1: 咖啡偏好 ---
  [user] 我最近喜欢上了精品咖啡,特别是埃塞俄比亚的耶加雪菲
  ...
  >>> 触发了记忆进化! 新增 3 条, 合并 0 条

--- 会话 2: 天气与心情 ---
  ...
  >>> 触发了记忆进化! 新增 2 条, 合并 1 条

--- 会话 3: 咖啡深度对话 ---
  ...
  >>> 触发了记忆进化! 新增 1 条, 合并 2 条

=== 记忆系统状态 ===
{
  "episodic_count": 15,
  "semantic_count": 5,
  "graph_nodes": 5,
  "graph_edges": 8,
  "consolidations_triggered": 3
}

=== 检索: 咖啡推荐 ===
1. [0.89] 用户偏好精品咖啡,尤其是耶加雪菲
2. [0.76] 用户喜欢手冲咖啡(V60)
3. [0.71] 用户乳糖不耐受,使用燕麦奶

关联图已导出到 ./data/graph.html
```

## 风险与局限

- **MVP 单进程**：未处理文件并发写入
- **Mock Embedding 区分度有限**：真实场景建议使用 OpenAI Embeddings
- **LLM 输出依赖**：真实 LLM 模式下需要稳定的 JSON schema 输出

## License

MIT
