# core/llm_client.py
import os
from typing import List, Optional, AsyncGenerator
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from core.types import Message, ToolCall, ToolCallFunction

# 加载.env文件（读取API密钥、地址）
load_dotenv()


class LLMResponse:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[ToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls


class LLMClient:
    def __init__(self):
        # 从.env文件中读取配置项
        self.base_url = os.getenv("LLM_BASE_URL")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL")

        # 初始化OpenAI异步客户端
        self.client = self._create_client()

        if not self.base_url or not self.api_key:
            raise RuntimeError("LLM_BASE_URL 和 LLM_API_KEY 必须在.env配置")

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
                        "arguments": tc.function.arguments
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
            arguments=func_dict["arguments"]
        )
        tool_call = ToolCall(
            id=raw_tool_call["id"],
            type=raw_tool_call["type"],
            function=func
        )
        return tool_call

    async def chat_stream(
            self,
            messages: List[Message],
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        流式接口：只输出纯文本，不处理toolcall
        只有invoke确认不需要工具时，才调用这个接口给前端打字机
        yield 字符串片段
        """
        openai_messages = [self._convert_msg_to_openai_dict(m) for m in messages]
        req_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "stream": True
        }
        response = await self.client.chat.completions.create(**req_params)

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content
            if text is not None:
                # yield：逐块输出文本
                yield text

    async def chat_invoke(
            self,
            messages: List[Message],
            tools_schema: Optional[List[dict]] = None,
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
        }

        # 2. 调用SDK接口
        response = await self.client.chat.completions.create(**req_params)

        # 3. 解析返回结果
        msg_raw = response.choices[0].message

        content = msg_raw.content
        parsed_tool_calls: Optional[List[ToolCall]] = None

        # 如果返回tool_calls，转换成我们自己的ToolCall对象
        if msg_raw.tool_calls:
            parsed_tool_calls = [self._convert_openai_tool_call(tc) for tc in msg_raw.tool_calls]

        return LLMResponse(content=content, tool_calls=parsed_tool_calls)
