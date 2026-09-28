# core/llm_client.py
import json
import logging
import os
from typing import List, Optional, AsyncGenerator

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from core.types import Message, ToolCall, ToolCallFunction, StopReason

# 加载.env文件（读取API密钥、地址）
load_dotenv()

logger = logging.getLogger(__name__)


class LLMResponse:
    def __init__(
            self, content: Optional[str],
            tool_calls: Optional[List[ToolCall]] = None,
            stop_reason: StopReason = StopReason.Stop,
            error_message: Optional[str] = None
    ):
        self.content = content
        self.tool_calls = tool_calls


class LLMClient:
    def __init__(self):
        # 从.env文件中读取配置项
        self.base_url = os.getenv("LLM_BASE_URL")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL")

        if not self.base_url or not self.api_key:
            raise RuntimeError("LLM_BASE_URL 和 LLM_API_KEY 必须在.env配置")

        # 初始化OpenAI异步客户端
        self.client = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _convert_msg_to_openai_dict(self, msg: Message) -> dict:
        """
        【私有方法】把我们自己的Message对象，转换成OpenAI协议的字典
        """
        item = {
            "role": msg.role,
        }
        if msg.content is not None:
            item["content"] = msg.content

        # assistant消息携带tool_calls
        if msg.role == "assistant" and msg.tool_calls is not None and len(msg.tool_calls) > 0:
            item["tool_calls"] = []
            for tc in msg.tool_calls:
                item["tool_calls"].append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        # OpenAI协议要求arguments是JSON字符串，回传历史时必须把dict序列化回字符串
                        "arguments": json.dumps(tc.function.arguments, ensure_ascii=False)
                    }
                })
        # tool消息携带tool_call_id
        if msg.role == "tool":
            item["tool_call_id"] = msg.tool_call_id

        return item

    def _convert_openai_tool_call(self, raw_tool_call: dict) -> ToolCall:
        """把OpenAI返回原始tool_call字典，转为我们框架的ToolCall对象"""
        func_dict = raw_tool_call["function"]
        func = ToolCallFunction(
            name=func_dict["name"],
            arguments=self._parse_arguments(func_dict["arguments"])
        )
        tool_call = ToolCall(
            id=raw_tool_call["id"],
            type=raw_tool_call["type"],
            function=func
        )
        return tool_call

    def _parse_arguments(self, raw) -> dict:
        """
        OpenAI返回的arguments是JSON字符串，转成dict
        框架内部统一用dict表示，只在收发两个协议边界做转换
        """
        if isinstance(raw, dict):
            # 部分兼容网关直接返回dict，兜底
            return raw
        if not raw:
            # 无参工具调用会返回空字符串
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # 坏JSON（如输出被token截断）：按空参处理，让工具报错、模型下轮自纠，而不是炸掉整个循环
            logger.warning("tool_call arguments不是合法JSON，已按空参数处理: %r", raw)
            return {}

    async def chat_invoke(
            self,
            messages: List[Message],
            tool_schemas: Optional[List[dict]] = None,
    ) -> LLMResponse:
        """
        非流式对话（先实现这个，最简单，方便调试）
        入参：我们框架的Message列表，工具schema数组
        返回：LLMResponse（包含文本 / tool_calls）
        """
        # 1. 框架内部Message 转 OpenAI需要的dict数组
        openai_messages = [self._convert_msg_to_openai_dict(m) for m in messages]

        req_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "tools": tool_schemas,
        }

        # 2. 调用SDK接口
        try:
            response = await self.client.chat.completions.create(**req_params)
        except openai.APIError as e:
            return LLMResponse(
                content=None,
                stop_reason=StopReason.Error,
                error_message=f"OpenAI API错误: {str(e.message)}"
            )

        # 3. 解析返回结果
        msg_raw = response.choices[0].message
        finish = response.choices[0].finish_reason

        content = msg_raw.content
        parsed_tool_calls: Optional[List[ToolCall]] = None

        # 如果返回tool_calls，转换成我们自己的ToolCall对象
        if msg_raw.tool_calls:
            parsed_tool_calls = [self._convert_openai_tool_call(tc) for tc in msg_raw.tool_calls]

        # 4. 映射成框架的StopReason（新增）
        if finish == "length":
            stop_reason = StopReason.Length  # 截断优先：即使带了tool_calls也不执行
        elif parsed_tool_calls:
            stop_reason = StopReason.ToolUse  # 兜底：部分网关finish_reason返回"stop"但确实有tool_calls
        else:
            stop_reason = StopReason.Stop

        return LLMResponse(content=content, tool_calls=parsed_tool_calls, stop_reason=stop_reason)
