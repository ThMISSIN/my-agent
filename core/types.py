# core/types.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uuid


class ToolCallFunction(BaseModel):
    """工具调用内的function结构体，对应LLM返回function"""
    name: str
    # default_factory=dict：每次新建实例都会生成全新空字典，避免共享引用问题
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """单次工具调用对象，OpenAI格式对齐"""
    # lambda：每次new对象自动生成一个唯一的id
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: str = "function"
    function: ToolCallFunction


class ToolResult(BaseModel):
    """工具执行完成后的返回结果，写入消息栈"""
    tool_call_id: str
    content: str
    # 可选标记：是否终止Agent循环（Pi支持提前退出）
    terminate: bool = False


class Message(BaseModel):
    """消息基类，对应会话中的一条消息"""
    # model_config是Pydantic v2 配置（写在类内部）
    # - "arbitrary_types_allowed": True：允许字段放自定义对象，关闭类型强校验限制。
    # - "extra": "ignore"：如果构造对象的时候传了 没有定义的字段 ，直接忽略，不抛异常。类似Java忽略未知字段。
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "ignore"
    }

    role: str = Field(..., description="system / user / assistant / tool")
    content: Optional[str] = Field(default=None, description="文本内容")
    # assistant角色专用：LLM输出的工具调用列表
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    # tool角色专用：关联的工具调用id
    tool_call_id: Optional[str] = Field(default=None)

    @classmethod
    def new_system_message(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def new_user_message(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def new_assistant_message(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def new_tool_message(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role="tool", tool_call_id=tool_call_id, content=content)
