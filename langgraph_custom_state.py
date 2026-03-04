# day3_custom_state.py
from __future__ import annotations

import json
from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# =========================
# 1) 自定义状态（Day 3 重点）
# =========================
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: NotRequired[str]         # weather | news | other
    city: NotRequired[str]           # 从用户问题里提取城市
    facts: NotRequired[list[str]]    # 中间信息沉淀
    retry_count: NotRequired[int]    # 预留给后续重试逻辑
    final_answer: NotRequired[str]   # 最终答案


# =========================
# 2) 工具定义（按节点拆分工具）
# =========================
@tool
def weather_lookup(city: str) -> str:
    """查询城市天气"""
    print(f"mark_weather_lookup: {city}")
    city_l = city.lower()
    if "上海" in city_l or "shanghai" in city_l:
        return "上海：30度，有雾。"
    if "北京" in city_l or "beijing" in city_l:
        return "北京：22度，晴。"
    return f"{city}：28度，多云。"


@tool
def news_lookup(city: str) -> str:
    """查询城市新闻"""
    return f"{city} 今日新闻摘要：地铁客流上涨，文旅活动增加。"


# =========================
# 3) 模型初始化
# =========================
ZHIPU_API_KEY="955fc756a3e941a6b62132b40cdd0b33.gat6V7zrA4IiLtes"
ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
# 1.初始化模型和工具，定义并绑定工具到模型
base_model = ChatOpenAI(
    model="glm-4-flash",   # 或 glm-4-plus / glm-4-air
    temperature=0,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
)

# 节点级别绑定工具（核心实践）
weather_model = base_model.bind_tools([weather_lookup])
news_model = base_model.bind_tools([news_lookup])
chat_model = base_model  # 普通闲聊节点不绑定工具


# =========================
# 4) 节点函数
# =========================
def classify_intent(state: AgentState):
    """识别意图 + 提取城市，写入自定义状态字段"""
    messages = state["messages"]
    user_text = messages[-1].content if messages else ""

    prompt = (
        "你是一个意图分类器。"
        "请严格返回 JSON，格式为："
        '{"intent":"weather|news|other","city":"若无则空字符串"}。'
        "其中，前面intent中的weather表示天气，news表示新闻，other表示其他。"
        "不要输出任何额外文字。"
    )
    resp = base_model.invoke([SystemMessage(content=prompt), HumanMessage(content=str(user_text))])

    intent = "other"
    city = ""
    try:
        print(f"resp.content:{resp.content}")
        data = json.loads(resp.content)
        intent = data.get("intent", "other")
        city = data.get("city", "")
        if not city:
            city = state.get("city", "")
    except Exception:
        # 解析失败时降级，后续依旧可走 other 分支
        pass

    print(f"mark_classify_intent intent:{intent} city:{city}")

    facts = state.get("facts", [])
    facts.append(f"intent={intent}, city={city or 'N/A'}")

    print(f"intent={intent}, city={city or 'N/A'}")

    return {
        "intent": intent,
        "city": city,
        "facts": facts,
        "retry_count": state.get("retry_count", 0),
    }


def weather_agent(state: AgentState):
    """天气节点：只允许调用 weather_lookup"""
    messages = state["messages"]
    last = messages[-1]
    city = state.get("city") or "上海"
    print("mark_weather_agent")

    # 关键：当上一步已经是工具结果时，直接总结回答，避免再次触发工具调用
    if isinstance(last, ToolMessage):
        resp = chat_model.invoke(
            messages
            + [HumanMessage(content="请基于上面的工具结果，直接给出简洁中文回答，不要再调用任何工具。")]
        )
    else:
        hint = HumanMessage(content=f"用户意图是天气，请优先调用工具查询：{city}")
        resp = weather_model.invoke(messages + [hint])
    return {"messages": [resp]}


def news_agent(state: AgentState):
    """新闻节点：只允许调用 news_lookup"""
    messages = state["messages"]
    last = messages[-1]
    city = state.get("city") or "上海"

    if isinstance(last, ToolMessage):
        resp = chat_model.invoke(
            messages
            + [HumanMessage(content="请基于上面的工具结果，直接给出简洁中文回答，不要再调用任何工具。")]
        )
    else:
        hint = HumanMessage(content=f"用户意图是新闻，请优先调用工具查询：{city}")
        resp = news_model.invoke(messages + [hint])
    return {"messages": [resp]}


def chat_agent(state: AgentState):
    """兜底闲聊节点"""
    resp = chat_model.invoke(state["messages"])
    return {"messages": [resp]}


def finalize(state: AgentState):
    """统一收口：写 final_answer + 补充 facts"""
    last = state["messages"][-1]
    answer = last.content if hasattr(last, "content") else str(last)

    facts = state.get("facts", [])
    facts.append("finalized")

    return {
        "final_answer": answer,
        "facts": facts,
    }


# =========================
# 5) 路由函数
# =========================
def route_by_intent(state: AgentState) -> Literal["weather_agent", "news_agent", "chat_agent"]:
    intent = state.get("intent", "other")
    if intent == "weather":
        return "weather_agent"
    if intent == "news":
        return "news_agent"
    return "chat_agent"


def route_after_weather(state: AgentState) -> Literal["weather_tools", "finalize"]:
    last = state["messages"][-1]
    print(f"mark_route_after_weather: {last}")
    if getattr(last, "tool_calls", None):
        return "weather_tools"
    return "finalize"


def route_after_news(state: AgentState) -> Literal["news_tools", "finalize"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "news_tools"
    return "finalize"


# =========================
# 6) 构图
# =========================
graph = StateGraph(AgentState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("news_agent", news_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("weather_tools", ToolNode([weather_lookup]))
graph.add_node("news_tools", ToolNode([news_lookup]))
graph.add_node("finalize", finalize)

graph.set_entry_point("classify_intent")
graph.add_conditional_edges("classify_intent", route_by_intent)

graph.add_conditional_edges("weather_agent", route_after_weather)
graph.add_edge("weather_tools", "weather_agent")

graph.add_conditional_edges("news_agent", route_after_news)
graph.add_edge("news_tools", "news_agent")

graph.add_edge("chat_agent", "finalize")
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=MemorySaver())


# =========================
# 7) 运行示例
# =========================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "day3-demo"}}

    # 第1轮
    state1 = app.invoke(
        {
            "messages": [HumanMessage(content="帮我看看上海今天天气")],
            "facts": [],
            "retry_count": 0,
        },
        config=config,
    )
    print("第1轮回答：", state1.get("final_answer"))
    print("第1轮状态：intent=", state1.get("intent"), "city=", state1.get("city"), "facts=", state1.get("facts"))

    # 第2轮（同 thread_id，保留上下文）
    state2 = app.invoke(
        {"messages": [HumanMessage(content="再给我这个城市的新闻")]},
        config=config,
    )
    print("第2轮回答：", state2.get("final_answer"))
    print("第2轮状态：intent=", state2.get("intent"), "city=", state2.get("city"), "facts=", state2.get("facts"))