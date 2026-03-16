# day6_human_in_loop.py
from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# =========================
# 1) 状态定义
# =========================
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # 业务状态
    pending_action: NotRequired[str]         # 待执行动作
    risk_level: NotRequired[str]             # low/high
    awaiting_approval: NotRequired[bool]     # 是否等待人工审批
    approval_decision: NotRequired[str]      # approve/reject/unknown
    final_answer: NotRequired[str]


# =========================
# 2) 节点函数
# =========================
def plan_or_review(state: AgentState):
    """
    第一次进来：解析用户请求，判断风险。
    若上次在等待审批：读取本轮用户“同意/拒绝”。
    """
    messages = state["messages"]
    user_text = str(messages[-1].content) if messages else ""

    # 如果正在等待审批，解析审批结果
    if state.get("awaiting_approval", False):
        t = user_text.lower()
        if any(k in t for k in ["同意", "批准", "approve", "yes", "y"]):
            return {"approval_decision": "approve"}
        if any(k in t for k in ["拒绝", "不同意", "reject", "no", "n"]):
            return {"approval_decision": "reject"}
        return {"approval_decision": "unknown"}

    # 正常规划（demo 用规则判断高风险）
    high_risk_keywords = ["转账", "删除", "注销", "打款", "付款"]
    is_high_risk = any(k in user_text for k in high_risk_keywords)

    return {
        "pending_action": user_text,
        "risk_level": "high" if is_high_risk else "low",
        "awaiting_approval": False,
        "approval_decision": "",
    }


def request_approval(state: AgentState):
    action = state.get("pending_action", "未知动作")
    msg = (
        f"该操作属于高风险：{action}\n"
        "请人工确认：回复“同意”继续，回复“拒绝”终止。"
    )
    return {
        "awaiting_approval": True,
        "messages": [AIMessage(content=msg)],
        "final_answer": msg,
    }


def ask_again(state: AgentState):
    msg = "未识别你的审批结果，请明确回复：同意 / 拒绝。"
    return {"messages": [AIMessage(content=msg)], "final_answer": msg}


def execute_action(state: AgentState):
    action = state.get("pending_action", "未知动作")
    msg = f"已执行：{action}"
    return {
        "awaiting_approval": False,
        "final_answer": msg,
        "messages": [AIMessage(content=msg)],
    }


def reject_action(state: AgentState):
    action = state.get("pending_action", "未知动作")
    msg = f"已拒绝执行：{action}"
    return {
        "awaiting_approval": False,
        "final_answer": msg,
        "messages": [AIMessage(content=msg)],
    }


# =========================
# 3) 路由
# =========================
def route_after_plan_or_review(
    state: AgentState,
) -> Literal["request_approval", "execute_action", "reject_action", "ask_again"]:
    # 分支A：处于审批阶段
    if state.get("awaiting_approval", False):
        decision = state.get("approval_decision", "")
        if decision == "approve":
            return "execute_action"
        if decision == "reject":
            return "reject_action"
        return "ask_again"

    # 分支B：正常规划后
    if state.get("risk_level") == "high":
        return "request_approval"
    return "execute_action"


# =========================
# 4) 构图
# =========================
graph = StateGraph(AgentState)

graph.add_node("plan_or_review", plan_or_review)
graph.add_node("request_approval", request_approval)
graph.add_node("ask_again", ask_again)
graph.add_node("execute_action", execute_action)
graph.add_node("reject_action", reject_action)

graph.set_entry_point("plan_or_review")
graph.add_conditional_edges("plan_or_review", route_after_plan_or_review)

# 审批提示发出后先结束，等待下一轮人工输入（同 thread_id）
graph.add_edge("request_approval", END)
graph.add_edge("ask_again", END)
graph.add_edge("execute_action", END)
graph.add_edge("reject_action", END)

app = graph.compile(checkpointer=MemorySaver())


# =========================
# 5) 运行示例
# =========================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "day6-approval-demo"}}

    # 第1轮：高风险请求 -> 触发审批
    s1 = app.invoke(
        {"messages": [HumanMessage(content="请帮我转账给供应商 5000 元")]},
        config=config,
    )
    print("第1轮:", s1.get("final_answer"))

    # 第2轮：人工批准 -> 执行动作
    s2 = app.invoke(
        {"messages": [HumanMessage(content="同意")]},
        config=config,
    )
    print("第2轮:", s2.get("final_answer"))