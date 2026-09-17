"""端到端演示 — 展示智能体记忆自动进化的完整链路。

三段会话 × 每段 5 轮：
1. 咖啡偏好（耶加雪菲、手冲、燕麦奶）
2. 天气与心情（雨天 + 咖啡组合）
3. 咖啡深化（冷萃、浅烘焙 — 触发合并/更新）

每段结束后打印"触发进化: 新增 X 条, 合并 Y 条"。
最后展示状态、检索、导出关联图。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evolution import EvolutionConfig
from src.memory_agent import EvolvingMemoryAgent


# ---------------------------------------------------------------------------
# 演示数据
# ---------------------------------------------------------------------------
SESSIONS = [
    {
        "topic": "咖啡偏好",
        "turns": [
            ("user", "我最近喜欢上了精品咖啡,特别是埃塞俄比亚的耶加雪菲"),
            ("assistant", "耶加雪菲确实不错,有花香和柑橘调"),
            ("user", "对,我喜欢手冲,V60 滤杯"),
            ("user", "对了,我乳糖不耐受,所以喝燕麦奶"),
            ("user", "我每天早上 8 点喝一杯"),
        ],
    },
    {
        "topic": "天气与心情",
        "turns": [
            ("user", "今天下雨,心情有点低落"),
            ("assistant", "雨天适合看书或听音乐"),
            ("user", "我最近在读《思考,快与慢》"),
            ("user", "我喜欢下雨天在家里喝咖啡看书"),
            ("assistant", "咖啡和雨天很配"),
        ],
    },
    {
        "topic": "咖啡深化",
        "turns": [
            ("user", "对了,我忘记说了,我也喜欢冷萃咖啡"),
            ("assistant", "冷萃和手冲风味不同"),
            ("user", "是的,冷萃更甜,适合夏天"),
            ("user", "我最喜欢的还是耶加雪菲,那个花香忘不了"),
            ("user", "不过现在我更喜欢浅烘焙的豆子"),
        ],
    },
]


def banner(text: str) -> None:
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main() -> None:
    banner("智能体记忆进化系统演示")

    # 1. 初始化（默认 mock LLM + mock embedding）
    print("\n[1] 初始化 EvolvingMemoryAgent ...")
    config = EvolutionConfig(
        base_dir="./data",
        consolidation_threshold=5,  # 每 5 轮触发
        similarity_threshold=0.45,  # demo 用较宽阈值以触发合并
    )
    agent = EvolvingMemoryAgent(config)
    print(f"  - LLM 模式: {agent.orchestrator.llm.__class__.__name__}")
    print(f"  - Embedding 模式: {agent.orchestrator.embedding.__class__.__name__}")

    # 重置保证 demo 干净
    agent.reset()
    agent = EvolvingMemoryAgent(config)  # 重建

    # 2. 三段会话
    session_id = "demo_session"
    for sess in SESSIONS:
        banner(f"会话: {sess['topic']}")
        for role, content in sess["turns"]:
            print(f"\n  [{role}] {content}")
            if role == "user":
                resp = agent.chat(content, session_id=session_id)
                print(f"  [assistant] {resp}")
            else:
                # assistant 也写入 episodic(让 chat() 已经写了)
                pass

        # 段末检查是否触发进化
        status = agent.status()
        last = status["consolidator"].get("last_result")
        if last and (last["added"] or last["updated"]):
            print(
                f"\n  >>> 触发了记忆进化! "
                f"新增 {last['added']} 条, "
                f"合并 {last['updated']} 条, "
                f"聚类 {last['clusters']} 簇"
            )
        else:
            print(f"\n  --- 等待更多对话触发进化 (当前 {status['episodic_count']} 轮) ---")

    # 3. 状态快照
    banner("记忆系统状态")
    print(json.dumps(agent.status(), indent=2, ensure_ascii=False))

    # 4. 检索演示
    banner("检索演示")

    queries = [
        "推荐适合我的咖啡",
        "下雨天适合做什么",
        "我有什么饮食限制",
    ]
    for q in queries:
        print(f"\n>>> Query: {q}")
        print("-" * 40)
        ctx = agent.retrieve(q)
        print(ctx)
        print("-" * 40)

    # 5. 冲突演示
    banner("冲突处理演示 (Recency-wins)")
    print("\n>>> 手动注入矛盾记忆: '我不喜欢咖啡了,改喝茶'")
    before = agent.semantic.count()
    agent.observe("user", "我不喜欢咖啡了,改喝茶", session_id="conflict_session")
    agent.force_consolidate()
    after = agent.semantic.count()
    print(f"  - semantic 数量: {before} -> {after}")
    print("  - 最新检索 '咖啡' 的结果:")
    print("  " + agent.retrieve("咖啡").replace("\n", "\n  "))

    # 6. 关联图导出
    banner("关联图导出")
    out_path = "./data/graph.html"
    ok = agent.export_graph_html(out_path)
    if ok:
        print(f"  ✓ 已导出到 {out_path}")
        if Path(out_path).exists():
            print(f"  文件大小: {Path(out_path).stat().st_size} bytes")
    else:
        print("  ! pyvis 未安装,跳过 HTML 导出(可选依赖)")

    # 7. 数据文件清单
    banner("数据文件清单")
    data_dir = Path("./data")
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                print(f"  {p}  ({size} bytes)")

    print("\n演示完成!")


if __name__ == "__main__":
    main()
