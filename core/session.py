# core/session.py
from typing import List, Optional
from uuid import uuid4
from core.types import Message


class Session:
    def __init__(self, session_id: Optional[str] = None, system_prompt: Optional[str] = None):
        self.session_id = session_id or str(uuid4())
        # 消息栈，保存完整对话历史
        self.messages: List[Message] = []
        if system_prompt:
            self.append_system_message(system_prompt)

    def append_user_message(self, content: str) -> None:
        """追加用户消息"""
        msg = Message.new_user_message(content=content)
        self.messages.append(msg)

    def append_assistant_message(
            self,
            content: Optional[str] = None,
            tool_calls: Optional[List] = None
    ) -> None:
        """追加assistant消息（文本回答 / 工具调用）"""
        msg = Message.new_assistant_message(content=content, tool_calls=tool_calls)
        self.messages.append(msg)

    def append_tool_message(self, tool_call_id: str, content: str) -> None:
        """追加工具返回结果消息"""
        msg = Message.new_tool_message(tool_call_id=tool_call_id, content=content)
        self.messages.append(msg)

    def append_system_message(self, content: str) -> None:
        """追加system系统提示词"""
        msg = Message.new_system_message(content=content)
        self.messages.append(msg)

    def clear_messages(self) -> None:
        """清空消息栈，保留session_id"""
        self.messages.clear()

    def snapshot(self) -> List[Message]:
        """生成消息快照：返回消息的拷贝，用于fork分支"""
        return [msg.model_copy() for msg in self.messages]

    def fork(self, new_session_id: Optional[str] = None) -> "Session":
        """
        Pi风格：fork分支会话
        基于当前会话快照新建一个独立session，用于尝试不同Agent执行路径，互不干扰
        """
        new_sess = Session(session_id=new_session_id)
        new_sess.messages = self.snapshot()
        return new_sess
