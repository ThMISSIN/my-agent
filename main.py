# main.py
"""
Agent 框架入口脚本

用法：
    python main.py                     # 交互式多轮对话（推荐）
    python main.py "帮我算一下订单"     # 一次性提问后退出
    python main.py --no-tools "你好"    # 不注册任何工具，纯对话
    python main.py --max-turns 5       # 调整单次任务的最大工具轮数

交互模式内置命令：
    /new      开启新会话（清空上下文，保留工具）
    /history  查看当前会话消息栈
    /tools    查看已注册的工具
    /exit     退出
"""
import argparse
import asyncio
import sys
from pathlib import Path

from tools.rag.embedding_model import EmbeddingModel
from tools.rag.ingest_tool import IngestDocumentTool
from tools.rag.mineru_client import MinerUClient
from tools.rag.rag_tool import RAGRetrievalTool
from tools.rag.vector_store import ChromaVectorStore

# 保证直接以脚本方式运行时也能找到项目根目录下的 core/tools 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.agent_loop import AgentLoop, SYSTEM_PROMPT
from core.llm_client import LLMClient
from core.session import Session
from tools.tool_registry import ToolRegistry


def build_agent(enable_tools: bool, max_turns: int):
    """组装 Agent：LLM客户端 + 工具注册器 + 循环"""
    llm_client = LLMClient()
    registry = ToolRegistry()
    if enable_tools:
        # 检索与入库共用同一套 embedding / 向量库，避免连到分叉的库
        embedding = EmbeddingModel()
        store = ChromaVectorStore()
        registry.register_list([
            RAGRetrievalTool(embedding, store),
            IngestDocumentTool(embedding, store, MinerUClient()),
        ])
    agent = AgentLoop(llm_client=llm_client, tool_registry=registry, max_turns=max_turns)
    return agent, registry


def print_new_messages(session: Session, start_index: int) -> None:
    """打印本轮新增的消息（工具调用过程 + 最终回答）"""
    for msg in session.messages[start_index:]:
        content = (msg.content or "").strip()
        if msg.role == "assistant" and msg.tool_calls:
            calls = "; ".join(
                f"{tc.function.name}({tc.function.arguments})" for tc in msg.tool_calls
            )
            print(f"[模型→工具] {calls}")
        elif msg.role == "tool":
            shown = content if len(content) <= 200 else content[:200] + "…"
            print(f"[工具结果] {shown}")
        elif msg.role == "assistant" and content:
            print(f"\n助手: {content}\n")


async def chat_once(agent: AgentLoop, session: Session, query: str) -> None:
    """执行一轮对话：以消息栈下标为界，只打印本次新增的消息"""
    start = len(session.messages)
    await agent.run_loop(session, query)
    print_new_messages(session, start)


HELP_TEXT = "命令: /new 新会话 | /history 消息栈 | /tools 已注册工具 | /exit 退出"


async def interactive(agent: AgentLoop, registry: ToolRegistry) -> None:
    """交互式多轮对话：同一个 Session 内上下文连续"""
    session = Session(system_prompt=SYSTEM_PROMPT)
    print(f"== Agent 交互模式 ==\n{HELP_TEXT}")
    while True:
        try:
            query = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        if query in ("/exit", "/quit", "/q"):
            print("再见！")
            break
        if query == "/new":
            session = Session(system_prompt=SYSTEM_PROMPT)
            print("已开启新会话。")
            continue
        if query == "/history":
            for i, m in enumerate(session.messages):
                text = m.content or ("(工具调用)" if m.tool_calls else "")
                print(f"{i:>3} [{m.role}] {text[:80]}")
            continue
        if query == "/tools":
            print("已注册工具:", ", ".join(registry.list_tool_names()) or "(无)")
            continue
        # 单轮出错不退出 REPL，方便继续调试
        try:
            await chat_once(agent, session, query)
        except Exception as e:
            print(f"[本轮执行出错] {type(e).__name__}: {e}")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Agent 入口脚本")
    parser.add_argument("query", nargs="*", help="一次性提问；不传则进入交互模式")
    parser.add_argument("--no-tools", action="store_true", help="不注册任何工具，纯对话")
    parser.add_argument("--max-turns", type=int, default=15, help="单次任务最大工具轮数，默认15")
    args = parser.parse_args()

    agent, registry = build_agent(enable_tools=not args.no_tools, max_turns=args.max_turns)

    if args.query:
        session = Session(system_prompt=SYSTEM_PROMPT)
        await chat_once(agent, session, " ".join(args.query))
    else:
        await interactive(agent, registry)


def main() -> None:
    # Windows 控制台中文输出兜底，避免 GBK 编码报错
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n已中断。")


if __name__ == "__main__":
    main()
