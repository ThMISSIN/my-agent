# tools/rag/ingest_tool.py
from typing import Any, Dict

from core.types import ToolResult
from tools.base_tool import BaseTool
from tools.rag.rag_ingest import ingest_path


class IngestDocumentTool(BaseTool):
    """
    文档入库工具：读取本地文件/目录 → 调用自研 MinerUClient(批量上传+轮询+取 full.md)
    解析 → 结构感知切块 → 向量化 → 写入知识库。注册进 ToolRegistry 即可被 Agent 调用。
    协议层只做入参校验与异常→ToolResult，具体流程在 rag_ingest。
    """

    def __init__(self, embedding_model, vector_store, mineru_client):
        self._embedding = embedding_model
        self._store = vector_store
        self._mineru = mineru_client

    @property
    def name(self) -> str:
        return "add_documents_to_kb"

    @property
    def description(self) -> str:
        return (
            "当用户要求把本地 PDF/Word/PPT/Excel/图片等文档加入知识库时调用。"
            "入参为本地文件或目录路径；工具会在线解析、切块、向量化并写入向量库。"
            "在线解析有延迟，入库可能较慢；同一文件重复入库默认覆盖(reindex)。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要入库的本地文件路径或目录路径（目录会递归按扩展名过滤）"
                },
                "reindex": {
                    "type": "boolean",
                    "description": "是否先删除同名文件的历史块再写入，默认 true"
                }
            },
            "required": ["path"]
        }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        path = (arguments.get("path") or "").strip()
        if not path:
            return ToolResult(content="缺少参数 path：请提供要入库的文件或目录路径", success=False)

        reindex = arguments.get("reindex", True)
        if isinstance(reindex, str):
            reindex = reindex.strip().lower() not in ("false", "0", "no", "")

        try:
            summary = await ingest_path(
                path=path,
                store=self._store,
                embedding=self._embedding,
                mineru=self._mineru,
                reindex=bool(reindex),
            )
        except FileNotFoundError as e:
            return ToolResult(content=str(e), success=False)
        except Exception as e:  # noqa: BLE001 兜底：任何意外都不炸循环
            return ToolResult(content=f"文档入库失败：{type(e).__name__}: {e}", success=False)

        # 全部失败视为业务失败，部分成功仍返回 success（明细已含失败原因）
        success = summary.files_failed == 0 or summary.files_ok > 0
        return ToolResult(content=summary.to_text(), success=success)
