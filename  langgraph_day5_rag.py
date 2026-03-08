# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# =========================
# 1) 知识库（示例）
# =========================
KB = [
    {
        "id": "doc_1",
        "title": "上海天气常识",
        "content": "上海春季多雨，夏季高温潮湿，秋季较舒适，冬季湿冷。",
    },
    {
        "id": "doc_2",
        "title": "北京天气常识",
        "content": "北京夏季炎热，冬季寒冷干燥，春秋季较短。",
    },
    {
        "id": "doc_3",
        "title": "上海地铁客流",
        "content": "工作日早晚高峰客流显著增加，周末旅游线路较繁忙。",
    },
    {
        "id": "doc_4",
        "title": "新闻摘要说明",
        "content": "新闻问答应优先基于最近事件与可靠来源，避免主观臆断。",
    },
]


# =========================
# 2) 状态定义
# =========================
class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: NotRequired[str]
    need_rag: NotRequired[bool]
    retrieved_docs: NotRequired[list[dict]]
    filtered_docs: NotRequired[list[dict]]
    final_answer: NotRequired[str]


# =========================
# 3) 模型初始化（按你的智谱配置）
# =========================
ZHIPU_API_KEY = "你的key"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

model = ChatOpenAI(
    model="glm-4-flash",
    temperature=0,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
)


# =========================
# 4) 辅助函数：简易检索（Day5先用轻量版）
# =========================
def _tokenize(text: str) -> list[str]:
    # 中英混合的极简分词：中文连续串 + 英文词
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())


def _score(query: str, doc_text: str) -> int:
    q_tokens = set(_tokenize(query))
    d_tokens = set(_tokenize(doc_text))
    return len(q_tokens.intersection(d_tokens))


# =========================
# 5) 节点函数
# =========================
def prepare_query(state: RAGState):
    messages = state["messages"]
    last_user = messages[-1].content if messages else ""
    return {"query": str(last_user)}


def judge_need_rag(state: RAGState):
    q = (state.get("query") or "").lower()
    # demo：包含这些词就走RAG；你可按业务扩展
    rag_keywords = ["天气", "新闻", "地铁", "资料", "根据", "查询", "weather", "news"]
    need = any(k in q for k in rag_keywords)
    return {"need_rag": need}


def route_need_rag(state: RAGState) -> Literal["retrieve_docs", "direct_chat"]:
    return "retrieve_docs" if state.get("need_rag", False) else "direct_chat"


def retrieve_docs(state: RAGState):
    q = state.get("query", "")
    scored = []
    for doc in KB:
        s = _score(q, f"{doc['title']} {doc['content']}")
        scored.append((s, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 取Top-3（分数>0）
    top_docs = [d for s, d in scored if s > 0][:3]
    return {"retrieved_docs": top_docs}


def filter_docs(state: RAGState):
    docs = state.get("retrieved_docs", [])
    # demo：这里先直接透传；你后续可加“相关性阈值/重排模型”
    return {"filtered_docs": docs}


def route_after_filter(state: RAGState) -> Literal["answer_with_context", "fallback_no_docs"]:
    docs = state.get("filtered_docs", [])
    return "answer_with_context" if docs else "fallback_no_docs"


def answer_with_context(state: RAGState):
    q = state.get("query", "")
    docs = state.get("filtered_docs", [])

    context_lines = []
    for d in docs:
        context_lines.append(f"[{d['id']}|{d['title']}] {d['content']}")
    context_text = "\n".join(context_lines)

    prompt = (
        "你是检索增强问答助手。\n"
        "请只基于给定资料回答，不要编造。\n"
        "若资料不足，请明确说“资料不足”。\n"
        "回答末尾附上引用ID列表，如：引用: doc_1, doc_3"
    )

    resp = model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"问题：{q}\n\n资料：\n{context_text}"),
        ]
    )
    return {"final_answer": str(resp.content)}


def fallback_no_docs(state: RAGState):
    q = state.get("query", "")
    return {
        "final_answer": f"我没有检索到可用资料来回答：{q}。请补充更具体的问题或提供知识库内容。"
    }


def direct_chat(state: RAGState):
    # 不需要RAG时，直接对话
    resp = model.invoke(state["messages"])
    return {"final_answer": str(resp.content)}


def finalize(state: RAGState):
    # 将最终答案追加到消息，方便多轮上下文保留
    answer = state.get("final_answer", "")
    return {"messages": [HumanMessage(content=f"[assistant_answer]{answer}")]}  # demo写法


# =========================
# 6) 构图
# =========================
graph = StateGraph(RAGState)

graph.add_node("prepare_query", prepare_query)
graph.add_node("judge_need_rag", judge_need_rag)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("filter_docs", filter_docs)
graph.add_node("answer_with_context", answer_with_context)
graph.add_node("fallback_no_docs", fallback_no_docs)
graph.add_node("direct_chat", direct_chat)
graph.add_node("finalize", finalize)

graph.set_entry_point("prepare_query")
graph.add_edge("prepare_query", "judge_need_rag")
graph.add_conditional_edges("judge_need_rag", route_need_rag)

graph.add_edge("retrieve_docs", "filter_docs")
graph.add_conditional_edges("filter_docs", route_after_filter)

graph.add_edge("answer_with_context", "finalize")
graph.add_edge("fallback_no_docs", "finalize")
graph.add_edge("direct_chat", "finalize")
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=MemorySaver())


# =========================
# 7) 运行示例
# =========================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "day5-rag-demo"}}

    s1 = app.invoke(
        {"messages": [HumanMessage(content="上海的天气特点是什么？")]},
        config=config,
    )
    print("第1轮:", s1.get("final_answer"))

    s2 = app.invoke(
        {"messages": [HumanMessage(content="上海地铁客流有什么特点？")]},
        config=config,
    )
    print("第2轮:", s2.get("final_answer"))