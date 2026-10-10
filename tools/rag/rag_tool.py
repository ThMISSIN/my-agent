from tools.base_tool import BaseTool
from tools.rag.embedding_model import EmbeddingModel
from tools.rag.vector_store import ChromaVectorStore
from core.types import ToolResult
from typing import Dict, Any


class RAGRetrievalTool(BaseTool):
    """
    RAG 知识库检索工具
    完全符合 BaseTool 规范，注册进 ToolRegistry 即可被 Agent 自动调用
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        top_k: int = 3
    ):
        self._embedding = embedding_model
        self._store = vector_store
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "retrieve_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "当用户问题涉及内部规范、设备参数、业务流程、历史案例等知识库内容时调用，"
            "检索相关文档片段用于辅助回答。输入为用户的查询语句，返回最相关的文档内容。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的知识库查询语句，建议使用完整问题或核心关键词"
                }
            },
            "required": ["query"]
        }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        query = arguments.get("query", "").strip()
        if not query:
            return ToolResult(
                content="查询内容不能为空",
                success=False,
            )

        try:
            # 1. 查询转向量
            query_vec = await self._embedding.encode_single(query)
            # 2. 向量检索
            docs = await self._store.search(query_vec, top_k=self._top_k)

            if not docs:
                return ToolResult(
                    content="知识库中未找到相关文档",
                    success=False,
                )

            # 3. 格式化结果，拼接成自然语言文本给 LLM 阅读
            parts = ["【知识库检索结果】"]
            for i, doc in enumerate(docs, 1):
                parts.append(f"\n--- 文档{i} 相似度:{doc['score']} ---")
                meta = doc["metadata"]
                source = meta.get("file_name") or meta.get("source", "未知来源")
                section = meta.get("section")
                if section:
                    source = f"{source} · {section}"
                parts.append(f"来源：{source}")
                parts.append(f"内容：{doc['content']}")

            return ToolResult(
                content="\n".join(parts),
                success=True,
            )

        except Exception as e:
            return ToolResult(
                content=f"知识库检索失败：{str(e)}",
                success=False,
            )
