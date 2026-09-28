from tools.base_tool import BaseTool
from core.types import ToolResult
import aiohttp


class QueryOrderTool(BaseTool):
    """示例：调用Java后端【查询订单】接口"""
    @property
    def name(self) -> str:
        return "query_order"

    @property
    def description(self) -> str:
        return "查询订单信息，传入order_id，返回订单详情"

    @property
    def parameters(self) -> dict:
        # JSON Schema，告诉大模型入参要求
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单编号"
                }
            },
            "required": ["order_id"]
        }

    async def run(self, arguments: dict) -> ToolResult:
        # 1. 取出LLM传过来的参数
        order_id = arguments.get("order_id")

        # 2. 请求你的Java后端接口
        java_api_url = "http://127.0.0.1:8080/api/order/query"
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=java_api_url,
                headers=headers,
                json={"order_id": order_id}
            ) as resp:
                resp_text = await resp.text()
                if resp.status != 200:
                    return ToolResult(
                        success=False,
                        content=f"Java后端接口调用失败，code={resp.status}, msg={resp_text}"
                    )
                # 返回Java接口原始结果
                return ToolResult(
                    success=True,
                    content=resp_text
                )
