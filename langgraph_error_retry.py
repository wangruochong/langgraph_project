# day4_retry_fallback.py
from __future__ import annotations

import json
from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# =========================
# 1) 自定义状态（Day4 重点）
# =========================
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: NotRequired[str]          # weather | news | other
    city: NotRequired[str]
    tool_result: NotRequired[str]
    last_error: NotRequired[str]
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    final_answer: NotRequired[str]


# =========================
# 2) 工具（带可控失败）
# =========================
@tool
def weather_lookup(city: str, retry_count: int = 0) -> str:
    """查询天气（演示：第一次可能超时，第二次成功）"""
    # 用确定性失败来方便学习重试流程：上海第一次必失败
    if city == "上海" and retry_count == 0:
        raise TimeoutError("天气服务超时（模拟）")
    return f"{city}：30度，微风。"


@tool
def news_lookup(city: str, retry_count: int = 0) -> str:
    """查询新闻（演示：可稳定成功）"""
    return f"{city} 今日新闻：地铁客流上涨，文旅活动增加。"


# =========================
# 3) 模型初始化
# =========================
ZHIPU_API_KEY = "955fc756a3e941a6b62132b40cdd0b33.gat6V7zrA4IiLtes"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

base_model = ChatOpenAI(
    model="glm-4-flash",
    temperature=0,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
)

classifier_model = ChatOpenAI(
    model="glm-4-flash",
    temperature=0,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    model_kwargs={"response_format": {"type": "json_object"}},
)


# =========================
# 4) 节点函数
# =========================
def classify_intent(state: AgentState):
    messages = state.get("messages", [])
    recent_messages = messages[-6:]  # 防止上下文无限增长
    prompt = (
        "你是意图分类器。只返回 JSON。"
        '格式: {"intent":"weather|news|other","city":"..."}。'
        'intent 只能是 weather/news/other。'
        'city 必须是具体城市名；若无法确定返回空字符串 ""。'
        "若最后一句出现“这个城市”等指代，请参考历史消息解析。"
    )

    resp = classifier_model.invoke([SystemMessage(content=prompt)] + recent_messages)

    prev_intent = state.get("intent", "other")
    prev_city = state.get("city", "")
    intent = prev_intent
    city = prev_city

    try:
        data = json.loads(str(resp.content).strip())

        intent_raw = str(data.get("intent", "")).strip().lower()
        if intent_raw in {"weather", "news", "other"}:
            intent = intent_raw
        else:
            last_text = str(messages[-1].content) if messages else ""
            if "新闻" in last_text or "news" in last_text.lower():
                intent = "news"
            elif "天气" in last_text or "weather" in last_text.lower():
                intent = "weather"
            else:
                intent = "other"

        city_raw = str(data.get("city", "")).strip()
        if city_raw.lower() in {"this_city", "that_city", "none", "null"} or city_raw in {"这个城市", "该城市"}:
            city_raw = ""
        city = city_raw or prev_city

    except Exception:
        # 解析失败也不让流程崩
        pass

    return {
        "intent": intent,
        "city": city,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "last_error": "",
    }


def execute_tool(state: AgentState):
    """统一执行工具，失败写入 last_error，不抛异常到图外"""
    intent = state.get("intent", "other")
    city = state.get("city", "") or "上海"
    retry_count = state.get("retry_count", 0)

    try:
        if intent == "weather":
            result = weather_lookup.invoke({"city": city, "retry_count": retry_count})
        elif intent == "news":
            result = news_lookup.invoke({"city": city, "retry_count": retry_count})
        else:
            result = ""
        return {"tool_result": result, "last_error": ""}
    except Exception as e:
        return {"tool_result": "", "last_error": str(e)}


def retry_handler(state: AgentState):
    return {"retry_count": state.get("retry_count", 0) + 1}


def fallback_answer(state: AgentState):
    intent = state.get("intent", "other")
    city = state.get("city", "该城市")
    err = state.get("last_error", "未知错误")
    answer = f"抱歉，{intent}查询暂时不可用（城市：{city}，错误：{err}）。请稍后重试。"
    return {"final_answer": answer}


def finalize(state: AgentState):
    tool_result = state.get("tool_result", "")
    intent = state.get("intent", "other")
    city = state.get("city", "该城市")
    answer = f"已为你完成{intent}查询（{city}）：{tool_result}"
    return {"final_answer": answer}


def chat_agent(state: AgentState):
    resp = base_model.invoke(state["messages"])
    return {"final_answer": str(resp.content)}


# =========================
# 5) 路由函数
# =========================
def route_by_intent(state: AgentState) -> Literal["execute_tool", "chat_agent"]:
    if state.get("intent") in {"weather", "news"}:
        return "execute_tool"
    return "chat_agent"


def route_after_execute(state: AgentState) -> Literal["finalize", "retry_handler", "fallback_answer"]:
    has_error = bool(state.get("last_error"))
    if not has_error:
        return "finalize"

    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "retry_handler"
    return "fallback_answer"


# =========================
# 6) 构图
# =========================
graph = StateGraph(AgentState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("execute_tool", execute_tool)
graph.add_node("retry_handler", retry_handler)
graph.add_node("fallback_answer", fallback_answer)
graph.add_node("finalize", finalize)
graph.add_node("chat_agent", chat_agent)

graph.set_entry_point("classify_intent")
graph.add_conditional_edges("classify_intent", route_by_intent)
graph.add_conditional_edges("execute_tool", route_after_execute)

graph.add_edge("retry_handler", "execute_tool")
graph.add_edge("fallback_answer", END)
graph.add_edge("finalize", END)
graph.add_edge("chat_agent", END)

app = graph.compile(checkpointer=MemorySaver())


# =========================
# 7) 运行示例
# =========================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "day4-demo"}}

    state = app.invoke(
        {
            "messages": [HumanMessage(content="帮我看下上海天气")],
            "retry_count": 0,
            "max_retries": 2,
        },
        config=config,
    )
    print("结果1:", state.get("final_answer"))
    print("状态1:", {"intent": state.get("intent"), "city": state.get("city"), "retry_count": state.get("retry_count"), "last_error": state.get("last_error")})

    state2 = app.invoke(
        {"messages": [HumanMessage(content="再给我这个城市的新闻")]},
        config=config,
    )
    print("结果2:", state2.get("final_answer"))
    print("状态2:", {"intent": state2.get("intent"), "city": state2.get("city"), "retry_count": state2.get("retry_count"), "last_error": state2.get("last_error")})