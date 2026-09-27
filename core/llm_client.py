# core/llm_client.py
import os
from typing import List, Optional
import aiohttp
from dotenv import load_dotenv

# 导入我们自己定义的数据模型
from core.types import Message, ToolCall, ToolCallFunction

# 加载.env文件（读取API密钥、地址）
load_dotenv()


# 定义LLM返回结果的包装对象（类比Java DTO）
class LLMResponse:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[ToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls


class LLMClient:
    def __init__(self):
        # 从环境变量读取配置，类比Java读取application.yml
        self.base_url: str = os.getenv("LLM_BASE_URL")
        self.api_key: str = os.getenv("LLM_API_KEY")
        self.model_name: str = os.getenv("LLM_MODEL", "deepseek-chat")

        if not self.base_url or not self.api_key:
            raise RuntimeError("LLM_BASE_URL 和 LLM_API_KEY 必须在.env配置")

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _convert_msg_to_openai_dict(self, msg: Message) -> dict:
        """
        【私有方法】把我们自己的Message对象，转换成OpenAI协议的字典
        Java类比：DTO转换器convert()
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

    async def chat_non_stream(
        self,
        messages: List[Message],
        tools_schema: Optional[List[dict]] = None
    ) -> LLMResponse:
        """
        非流式对话（先实现这个，最简单，方便调试）
        入参：我们框架的Message列表，工具schema数组
        返回：LLMResponse（包含文本 / tool_calls）
        """
        # 1. 将我们的Message列表，转成OpenAI协议需要的dict数组
        openai_messages = [self._convert_msg_to_openai_dict(m) for m in messages]

        req_body = {
            "model": self.model_name,
            "messages": openai_messages
        }
        # 如果传入工具定义，则加上tools字段，开启function call
        if tools_schema and len(tools_schema) > 0:
            req_body["tools"] = tools_schema
            req_body["tool_choice"] = "auto"

        # 2. 发起异步http请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=req_body
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"LLM请求失败 status={resp.status}, msg={text}")
                resp_json = await resp.json()

        # 3. 解析返回结果
        choice = resp_json["choices"][0]
        message_raw = choice["message"]

        content = message_raw.get("content")
        raw_tool_calls = message_raw.get("tool_calls")

        parsed_tool_calls: Optional[List[ToolCall]] = None
        if raw_tool_calls is not None:
            parsed_tool_calls = [self._convert_openai_tool_call(tc) for tc in raw_tool_calls]

        return LLMResponse(content=content, tool_calls=parsed_tool_calls)
