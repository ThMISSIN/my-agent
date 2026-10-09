from typing import Optional, List

from core.llm_client import LLMClient
from core.session import Session
from core.types import Message, MessageRole, StopReason
from tools.tool_registry import ToolRegistry

SYSTEM_PROMPT = """
# 智能助手提示词

你是一个智能助手，你的任务是帮助用户解决问题。你可以调用工具来获取信息。如果你的信息不足时，可以通过 rag 工具去调用外部的知识库。

请根据用户问题的类型，选择合适的信息获取方式：

## 1. 电网相关数据查询

当用户询问电网相关数据时，例如设备台账、运行数据、统计指标、实时/历史量测数据、告警记录、指标查询等，应优先调用 Java 后端接口进行数据查询。

- 调用 Java 后端接口时，应明确查询对象、时间范围、设备范围、指标类型等必要参数。
- 如果用户未提供必要参数，应先向用户询问缺失信息，或根据上下文合理补全后再调用。
- 如果接口返回结果为空、异常或权限不足，应向用户说明情况，并给出下一步建议。

## 2. 电网专业问题解答

当用户询问电网专业问题时，例如“三比值是什么”“局部放电有哪些类型”“变压器油色谱分析原理是什么”等概念性、原理性、标准规范类问题，应通过 rag 工具检索已经准备好的知识库进行学习解答。

- 回答时应优先依据知识库内容，不要凭空编造。
- 如果知识库中没有相关内容，应明确告知用户未检索到可靠资料，并建议用户补充问题背景或提供更多线索。

## 3. 组合使用工具

当用户的问题同时涉及数据查询和专业解释时，可以组合使用工具：

- 先调用 Java 后端接口获取相关数据；
- 再通过 rag 工具检索相关专业知识；
- 最后结合数据和知识库内容，给出完整、准确、易懂的解答。

## 4. 调用工具前的判断

在调用工具前，应尽量判断用户意图，避免无效调用。

- 如果问题模糊，无法判断是数据查询还是专业问答，应先向用户确认。
- 如果问题可以直接回答且不需要外部信息，可以不调用工具。

## 5. 回答要求

回答时应做到：

- **准确**：数据以接口返回为准，专业知识以知识库为准。
- **清晰**：说明数据来源或知识来源。
- **谨慎**：不要编造不存在的数据、接口、标准或结论。
- **有用**：尽量给出结论、解释和下一步建议。

## 目标

你的目标是：在电网场景下，帮助用户完成数据查询、专业问答和综合分析，并在信息不足时主动调用合适工具获取信息。
"""


class AgentLoop:
    """智能体循环，管理智能体的运行流程"""

    def __init__(
            self,
            llm_client: LLMClient,
            tool_registry: Optional[ToolRegistry] = None,
            max_turns: int = 15,
            system_prompt: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_turns = max_turns
        # 系统提示词由AgentLoop接管：不传时默认使用内置SYSTEM_PROMPT，传自定义串则覆盖
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    async def run_loop(self, session: Session, query: str):
        """运行智能体循环"""
        new_messages: List[Message] = [] # 本次运行新增的消息
        # 会话里没有system消息时自动补上，避免调用方漏配导致裸会话直接跑
        if self.system_prompt:
            sys_msg = session.ensure_system_message(self.system_prompt)
            if sys_msg:
                new_messages.append(sys_msg)
        # 将消息添加到会话记录中
        user_msg = session.append_user_message(query)

        new_messages.append(user_msg)

        turn = 0
        while True:
            turn += 1
            if turn > self.max_turns:
                tail = session.append_assistant_message(content="已达到最大轮数限制,本次任务终止")
                new_messages.append(tail)
                return new_messages

            # 获取tool列表
            tool_schemas = self.tool_registry.get_all_tool_schemas() if self.tool_registry else None

            # 调用LLM
            resp_message = await self.llm_client.chat_invoke(
                session.messages,
                tool_schemas=tool_schemas,
            )

            if resp_message.stop_reason == StopReason.Error:
                err = session.append_assistant_message(content=f"模型调用失败：{resp_message.error_message}")
                new_messages.append(err)
                return new_messages

            # ① 先入栈assistant消息（必须先于工具执行：协议要求tool消息紧跟在
            #    带tool_calls的assistant消息之后，顺序错了下次请求400）
            assistant_msg = session.append_assistant_message(content=resp_message.content, tool_calls=resp_message.tool_calls)
            new_messages.append(assistant_msg)

            # ② 终止条件：没有工具调用 → 模型给的就是最终答案，本次运行结束
            if not resp_message.tool_calls:
                return new_messages

            # ③ length截断保护（对齐 pi 的 failToolCallsFromTruncatedMessage）：
            #    被截断的消息里工具参数很可能是残缺JSON，全部置失败、不执行
            if resp_message.stop_reason == StopReason.Length:
                for tc in resp_message.tool_calls:
                    fail = session.append_tool_message(tc.id, "工具调用因输出token上限被截断，参数不完整，未执行")
                    new_messages.append(fail)
                continue  # 模型看到失败说明后会重新组织调用

            # ④ 执行全部工具调用，结果入栈，回到循环顶部让模型消费结果
            #    保护：没注册任何工具时模型不应返回tool_calls（网关异常/幻觉才会发生），
            #    统一置失败喂回让模型改走纯文本，而不是对None调方法崩溃；持续幻觉由max_turns兜底
            if self.tool_registry is None:
                for tc in resp_message.tool_calls:
                    fail = session.append_tool_message(tc.id, "工具调用失败：当前Agent未配置任何可用工具")
                    new_messages.append(fail)
                continue

            tool_msgs = await self.tool_registry.execute_tool_calls(resp_message.tool_calls)
            session.messages.extend(tool_msgs)
            new_messages.extend(tool_msgs)