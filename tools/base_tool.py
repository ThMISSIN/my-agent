from abc import ABC, abstractmethod

from typing import Dict, Any, List

from core.types import ToolResult


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一名称，LLM识别用，例如: query_user_info"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，告诉大模型这个工具是干嘛的"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        参数JSON Schema，给LLM的。
        对应OpenAI tools参数定义，告诉大模型调用该工具需要哪些参数
        """
        ...

    # ========= 核心执行方法，子类必须实现 =========
    @abstractmethod
    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        """
        执行工具逻辑
        :param arguments: LLM生成的参数字典
        :return: ToolCallResult（types中定义的返回对象）
        """
        ...

    def to_openai_tool_schema(self) -> Dict[str, Any]:
        """
        转为OpenAI要求的tools schema格式
        LLMClient.invoke的时候，收集所有注册工具，调用这个方法打包传给大模型
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
