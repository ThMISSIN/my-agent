# core/types.py
from enum import StrEnum

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
import uuid

class MessageRole(StrEnum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"

class StopReason(StrEnum):
    Stop = "stop"          # 正常结束（这就是最终答案）
    ToolUse = "tool_use"   # 模型请求调用工具
    Length = "length"      # 被输出token上限截断
    Error = "error"        # 模型侧/网络侧错误
    Aborted = "aborted"    # 被取消（初期先定义，可不实现触发路径）

class ToolCallFunction(BaseModel):
    """工具调用内的function结构体，LLM 输出的 function 调用信息"""
    name: str
    # default_factory=dict：每次新建实例都会生成全新空字典，避免共享引用问题
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """
    一次完整的工具调用请求，OpenAI格式对齐
    LLM 输出 → 解析成ToolCall对象 → Agent 拿到后执行工具。
    """
    # lambda：每次new对象自动生成一个唯一的id
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: str = "function"
    function: ToolCallFunction


class ToolResult(BaseModel):
    """工具执行完之后，返回给 LLM 的结果消息体，直接入 session 消息栈"""
    tool_call_id: str
    content: str
    success: bool = False


class Message(BaseModel):
    """消息基类，对应会话中的一条消息"""
    # model_config是Pydantic v2 配置（写在类内部）
    # - "arbitrary_types_allowed": True：允许字段放自定义对象，关闭类型强校验限制。
    # - "extra": "ignore"：如果构造对象的时候传了 没有定义的字段 ，直接忽略，不抛异常。类似Java忽略未知字段。
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "ignore"
    }

    role: MessageRole = Field(..., description="system / user / assistant / tool")
    content: Optional[str] = Field(default=None, description="文本内容")
    # assistant角色专用：LLM输出的工具调用列表
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    # tool角色专用：关联的工具调用id
    tool_call_id: Optional[str] = Field(default=None)

    @classmethod
    def new_system_message(cls, content: str) -> "Message":
        return cls(role=MessageRole.System, content=content)

    @classmethod
    def new_user_message(cls, content: str) -> "Message":
        return cls(role=MessageRole.User, content=content)

    @classmethod
    def new_assistant_message(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> "Message":
        return cls(role=MessageRole.Assistant, content=content, tool_calls=tool_calls)

    @classmethod
    def new_tool_message(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role=MessageRole.Tool, tool_call_id=tool_call_id, content=content)
