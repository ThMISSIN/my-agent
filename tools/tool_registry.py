import asyncio
from typing import Dict, List, Optional

from core.types import Message, ToolCall
from tools.base_tool import BaseTool


class ToolRegistry:
    """工具注册器，管理所有工具"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self,tool: BaseTool) -> None:
        """注册一个工具"""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already already registered")
        self._tools[tool.name] = tool

    def register_list(self, tool_list: List[BaseTool]) -> None:
        """批量注册多个工具"""
        for tool in tool_list:
            self.register(tool)

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """根据工具名字获取工具实例，找不到返回None"""
        return self._tools.get(tool_name)

    def get_all_openai_schemas(self) -> List[dict]:
        """
        获取全部工具的OpenAI schema数组
        在 LLMClient.invoke 的时候，直接把这个传给tools参数
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append(tool.to_openai_tool_schema())
        return schemas

    def list_tool_names(self) -> List[str]:
        """获取所有注册工具名，调试用"""
        return list(self._tools.keys())

    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        """
        执行一批工具调用，返回与tool_calls一一对应的tool消息。
        约定：任何单个工具失败都不抛异常——错误也必须变成tool消息喂回模型，
        模型才能自己修正参数重试或向用户解释。
        """
        messages: List[Message] = []
        for tc in tool_calls:  # 初期串行；后期I/O密集时可换 asyncio.gather 并发
            tool = self._tools.get(tc.function.name)
            if tool is None:
                # 模型幻觉出未注册的工具名是常态，报错并告知可用工具，让模型下轮自纠
                messages.append(Message.new_tool_message(
                    tc.id,
                    f"工具调用失败：未注册的工具 '{tc.function.name}'，可用工具：{self.list_tool_names()}"))
                continue
            try:
                result = await tool.run(tc.function.arguments)
                # success=False 时在内容里显式标注失败——OpenAI协议的tool消息没有错误字段，
                # 模型只能靠内容本身知道"这次失败了"
                content = result.content if result.success else f"工具执行失败：{result.content}"
            except asyncio.CancelledError:
                raise  # 取消信号必须继续向上传播，绝不能被 except Exception 吞掉
            except Exception as e:
                content = f"工具执行异常：{type(e).__name__}: {e}"
            messages.append(Message.new_tool_message(tc.id, content))
        return messages