#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   reasoning_agent_graph.py
@Time    :   2026/03/15 22:30:00
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   支持LLM自主判断并调用工具的推理代理（LangGraph工作流实现）
"""

import sys
import os
import json
import ast
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Optional, TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from agent.retriever import run_rag_engine 
from tools.tavily_search import TavilySearchTool
from tools.weather import WeatherTool


def load_env_from_root():
    """从项目根目录加载.env文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print(f"✅ 找到.env文件：{env_path}")
            return
        current_dir = os.path.dirname(current_dir)
    print("❌ 未找到.env文件")

load_env_from_root()

# 工具定义
TOOL_DEFINITIONS = {
    "web_search": {
        "description": "当需要获取最新/实时的公开信息、互联网数据时调用，例如：最新种植技术、市场价格、政策法规等",
        "func": TavilySearchTool().run
    },
    "weather": {
        "description": "当需要查询特定地区的天气/气候数据时调用，例如：温度、湿度、降雨量、季节气候等",
        "func": WeatherTool().run
    }
}


# ========== 状态定义 ==========
class ReasoningState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str                          # 用户原始问题
    retrieval_plan: Dict                 # 检索规划
    rag_result: Optional[Dict]           # RAG引擎返回结果
    tool_decision: List[str]             # 需要调用的工具列表
    tool_results: List[Dict]             # 已执行的工具结果列表
    current_tool_index: int               # 当前正在执行的工具索引
    all_results: List[Dict]               # 所有源结果（用于融合）
    fused_results: List[Dict]             # 融合后的结果
    optimized_answer: Optional[str]       # 最终优化答案
    error: Optional[str]                   # 错误信息
    step: str                              # 当前步骤


# ========== 全局工具实例（避免序列化问题）==========
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        # 从RAG引擎导入LLM或重新创建
        from rag.LLM import OpenAIChat
        _llm_instance = OpenAIChat()
    return _llm_instance


# ========== 工具函数 ==========
def fuse_multi_source_results(results_list: List[Dict]) -> List[Dict]:
    """
    融合多源结果（与原始类中的方法一致）
    """
    fused = []
    content_set = set()
    source_priority = {"vector": 0, "bm25": 1, "weather": 2, "web": 3}

    for res in results_list:
        if res.get("success") and res.get("data"):
            for item in res["data"]:
                content = item.get("content", "")
                if isinstance(content, (dict, list)):
                    content = str(content)
                content = content.strip() if isinstance(content, str) else ""
                if not content or content in content_set:
                    continue

                content_set.add(content)
                fused.append({
                    "content": content,
                    "source": item.get("source", "unknown"),
                    "priority": source_priority.get(item.get("source"), 99),
                })

    fused.sort(key=lambda x: x["priority"])
    return fused


def get_plan_config(retrieval_plan: Dict) -> Dict:
    if isinstance(retrieval_plan, dict):
        return retrieval_plan.get("data", retrieval_plan)
    return {}


def optimize_answer(
    raw_answer: str,
    query: str,
    context: List[str],
    detailed: bool = False,
    structured: bool = False,
    prefer_high_quality_context: bool = False
) -> str:
    """
    优化答案（与原始类中的方法一致）
    """
    context_to_use = context[:5] if prefer_high_quality_context else context
    context_text = "\n".join(context_to_use)
    extra_requirements = []
    if detailed:
        extra_requirements.append("尽量补充关键条件、注意事项和常见误区")
    if structured:
        extra_requirements.append("优先使用分点或分步骤表达，便于直接执行")

    prompt = """
优化以下食用菌种植问题的回答，要求：
1. 语言简洁专业，符合种植户阅读习惯
2. 补充必要的种植建议（基于上下文）
3. 避免重复和冗余
4. {extra_requirements}

原始回答：{raw_answer}
问题：{query}
上下文：{context}
优化后的回答：
""".format(
        raw_answer=raw_answer,
        query=query,
        context=context_text,
        extra_requirements="；".join(extra_requirements) if extra_requirements else "若上下文不足，请只保留有依据的内容"
    )

    llm = get_llm()
    optimized_response = llm.chat(
        prompt=prompt,
        history=[],
        content=context_text
    )
    return optimized_response.strip()


def extract_city_for_weather(query: str) -> str:
    """
    从查询中提取城市名称（用于天气工具）
    """
    city_extract_prompt = f"""
任务：从以下用户查询中提取需要查询天气的城市名称
要求：
1. 仅返回城市名称（如"北京"），无需任何解释、标点或额外文字
2. 若未提及具体城市，返回"未知"
3. 城市名称标准化（如"上海市"→"上海"，"北京朝阳区"→"北京"）

用户查询：{query}
城市名称：
"""
    llm = get_llm()
    city_name = llm.chat(
        prompt=city_extract_prompt,
        history=[],
        content=query
    ).strip()

    if not city_name or city_name in ["未知", "无", ""]:
        city_name = "未知"
    else:
        city_name = city_name.replace("市", "").replace("省", "").replace("区", "").replace("县", "")
    return city_name


# ========== 节点函数 ==========
def execute_rag_node(state: ReasoningState) -> dict:
    """节点1：执行RAG引擎的交织推理-检索循环（使用LangGraph实现的RAG引擎）"""
    query = state["query"]
    retrieval_plan = state["retrieval_plan"]

    try:
        # 调用新写的LangGraph RAG引擎
        rag_result = run_rag_engine(query, retrieval_plan)

        # 准备RAG数据格式，供后续融合
        # 注意：run_rag_engine返回的格式需要适配
        rag_data = {
            "success": rag_result.get("success", False),
            "data": [{"content": r.get("content", ""), "source": r.get("source", "rag")}
                     for r in rag_result.get("retrieval_results", [])]
        }

        return {
            "rag_result": rag_result,
            "all_results": [rag_data],
            "tool_results": [],
            "current_tool_index": 0,
            "step": "rag_completed",
            "messages": [AIMessage(content="RAG检索完成")]
        }
    except Exception as e:
        return {
            "error": f"RAG执行失败：{str(e)}",
            "step": "rag_failed",
            "messages": [AIMessage(content=f"❌ RAG执行失败：{str(e)}")]
        }


def judge_tool_node(state: ReasoningState) -> dict:
    """节点2：判断是否需要调用工具"""
    query = state["query"]
    rag_result = state["rag_result"]

    retrieval_plan = state.get("retrieval_plan", {})
    plan_config = get_plan_config(retrieval_plan)

    if not rag_result:
        return {
            "tool_decision": [],
            "step": "judge_skipped",
            "messages": [AIMessage(content="无RAG结果，跳过工具判断")]
        }

    # 获取最终答案，用于摘要
    final_answer = rag_result.get("final_answer", "")
    
    if plan_config.get("force_tool_call"):
        suggested_tools = [
            tool_name for tool_name in plan_config.get("suggested_tools", [])
            if tool_name in TOOL_DEFINITIONS
        ]
        if not suggested_tools:
            suggested_tools = ["web_search"]
        if suggested_tools:
            return {
                "tool_decision": suggested_tools,
                "step": "judge_completed",
                "messages": [AIMessage(content=f"根据优化策略强制调用工具：{suggested_tools}")]
            }

    tool_prompt = f"""
你需要分析以下查询和现有RAG检索结果，判断是否需要调用外部工具，规则如下：
1. 仅当RAG结果无法满足查询需求（如缺乏实时/最新数据、特定地区数据）时调用工具
2. 可调用工具列表：
{[f"{name}: {desc['description']}" for name, desc in TOOL_DEFINITIONS.items()]}
3. 输出格式：仅返回工具名称列表（如["weather"]或[]），无需额外解释

用户查询：{query}
RAG检索结果摘要：{final_answer[:500] if final_answer else "无检索结果"}
需要调用的工具列表：
"""
    llm = get_llm()
    tool_decision_raw = llm.chat(
        prompt=tool_prompt,
        history=[],
        content=final_answer
    ).strip()

    # 解析工具列表
    try:
        tool_list = ast.literal_eval(tool_decision_raw)
        tool_list = [t for t in tool_list if t in TOOL_DEFINITIONS]
    except:
        tool_list = []

    return {
        "tool_decision": tool_list,
        "step": "judge_completed",
        "messages": [AIMessage(content=f"工具决策完成：{tool_list}")]
    }


def execute_tool_node(state: ReasoningState) -> dict:
    """节点3：执行当前工具（根据索引）"""
    query = state["query"]
    tool_decision = state["tool_decision"]
    current_index = state.get("current_tool_index", 0)

    if current_index >= len(tool_decision):
        # 所有工具已执行完，跳转到融合
        return {
            "step": "all_tools_done",
            "messages": [AIMessage(content="所有工具执行完毕")]
        }

    tool_name = tool_decision[current_index]
    try:
        tool_func = TOOL_DEFINITIONS[tool_name]["func"]
        tool_input = query  # 默认输入

        if tool_name == "weather":
            # 天气工具需要提取城市
            city = extract_city_for_weather(query)
            tool_input = city

        # 执行工具
        tool_result = tool_func(tool_input)

        result_data = [{
            "content": tool_result,
            "source": tool_name,
        }]

        tool_res = {
            "success": True,
            "data": result_data,
            "error": None
        }

        # 更新状态
        new_tool_results = state.get("tool_results", []) + [tool_res]
        new_all_results = state.get("all_results", []) + [tool_res]

        return {
            "tool_results": new_tool_results,
            "all_results": new_all_results,
            "current_tool_index": current_index + 1,
            "step": f"tool_{tool_name}_done",
            "messages": [AIMessage(content=f"工具 {tool_name} 执行完成")]
        }

    except Exception as e:
        error_res = {
            "success": False,
            "data": [],
            "error": str(e)
        }
        new_tool_results = state.get("tool_results", []) + [error_res]
        new_all_results = state.get("all_results", []) + [error_res]

        return {
            "tool_results": new_tool_results,
            "all_results": new_all_results,
            "current_tool_index": current_index + 1,
            "error": f"工具 {tool_name} 执行失败：{str(e)}",
            "step": f"tool_{tool_name}_failed",
            "messages": [AIMessage(content=f"❌ 工具 {tool_name} 执行失败")]
        }


def fuse_results_node(state: ReasoningState) -> dict:
    """节点4：融合所有结果"""
    all_results = state.get("all_results", [])
    if not all_results:
        fused = []
    else:
        fused = fuse_multi_source_results(all_results)

    return {
        "fused_results": fused,
        "step": "fused",
        "messages": [AIMessage(content="结果融合完成")]
    }


def optimize_answer_node(state: ReasoningState) -> dict:
    """节点5：优化答案"""
    rag_result = state["rag_result"]
    query = state["query"]
    fused_results = state.get("fused_results", [])
    retrieval_plan = state.get("retrieval_plan", {})
    plan_config = get_plan_config(retrieval_plan)

    if not rag_result:
        optimized = "无法生成答案"
    else:
        context = [r["content"] for r in fused_results]
        optimized = optimize_answer(
            raw_answer=rag_result.get("final_answer", ""),
            query=query,
            context=context,
            detailed=plan_config.get("detailed_optimization", False),
            structured=plan_config.get("structured_answer", False),
            prefer_high_quality_context=plan_config.get("prefer_high_quality_context", False)
        )

    return {
        "optimized_answer": optimized,
        "step": "optimized",
        "messages": [AIMessage(content="答案优化完成")]
    }


def finalize_node(state: ReasoningState) -> dict:
    """节点6：整理最终结果"""
    return {
        "step": "completed",
        "messages": [AIMessage(content="推理代理工作流完成")]
    }


def error_handler_node(state: ReasoningState) -> dict:
    """错误处理节点"""
    error = state.get("error", "未知错误")
    return {
        "optimized_answer": f"处理失败：{error}",
        "step": "error_handled",
        "messages": [AIMessage(content=f"❌ 处理出错：{error}")]
    }


# ========== 条件边函数 ==========
def after_judge(state: ReasoningState) -> str:
    """工具判断后的路由：如果有工具则执行第一个工具，否则跳转到融合"""
    tool_decision = state.get("tool_decision", [])
    if tool_decision:
        return "execute_tool"
    else:
        return "fuse_results"


def after_tool(state: ReasoningState) -> str:
    """工具执行后的路由：检查是否还有未执行的工具"""
    tool_decision = state.get("tool_decision", [])
    current_index = state.get("current_tool_index", 0)
    if current_index < len(tool_decision):
        return "execute_tool"  # 继续执行下一个工具
    else:
        return "fuse_results"  # 所有工具完成，融合


# ========== 构建工作流图 ==========
def create_reasoning_agent():
    workflow = StateGraph(ReasoningState)

    # 添加节点
    workflow.add_node("execute_rag", execute_rag_node)
    workflow.add_node("judge_tool", judge_tool_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("fuse_results", fuse_results_node)
    workflow.add_node("optimize_answer", optimize_answer_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)

    # 设置边
    workflow.add_edge(START, "execute_rag")

    # RAG后判断错误
    workflow.add_conditional_edges(
        "execute_rag",
        lambda state: "error_handler" if state.get("error") else "judge_tool",
        {
            "judge_tool": "judge_tool",
            "error_handler": "error_handler"
        }
    )

    # 工具判断后路由
    workflow.add_conditional_edges(
        "judge_tool",
        after_judge,
        {
            "execute_tool": "execute_tool",
            "fuse_results": "fuse_results"
        }
    )

    # 工具执行后循环或结束
    workflow.add_conditional_edges(
        "execute_tool",
        after_tool,
        {
            "execute_tool": "execute_tool",
            "fuse_results": "fuse_results",
            "error_handler": "error_handler"  # 如果执行失败可进入错误处理
        }
    )

    # 融合后优化
    workflow.add_edge("fuse_results", "optimize_answer")

    # 优化后结束
    workflow.add_edge("optimize_answer", "finalize")
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# ========== 对外接口 ==========
class ReasoningGraphAgent:
    def __init__(self):
        self.app = create_reasoning_agent()

    def run(self, query: str, retrieval_plan: Dict) -> Dict:
        thread_id = f"reasoning-{hash(query)}-{hash(str(retrieval_plan))}"
        inputs = {
            "messages": [HumanMessage(content=f"开始处理：{query}")],
            "query": query,
            "retrieval_plan": retrieval_plan,
            "step": "start"
        }

        try:
            result = self.app.invoke(
                inputs,
                config={"configurable": {"thread_id": thread_id}}
            )

            return {
                "success": not result.get("error"),
                "fused_results": result.get("fused_results", []),
                "raw_answer": result.get("rag_result", {}).get("final_answer", ""),
                "optimized_answer": result.get("optimized_answer", ""),
                "retrieve_rounds": result.get("rag_result", {}).get("retrieve_rounds", 0),
                "called_tools": result.get("tool_decision", []),
                "tool_results": result.get("tool_results", []),
                "error": result.get("error")
            }
        except Exception as e:
            return {
                "success": False,
                "fused_results": [],
                "raw_answer": "",
                "optimized_answer": f"工作流执行失败：{str(e)}",
                "retrieve_rounds": 0,
                "called_tools": [],
                "tool_results": [],
                "error": str(e)
            }


if __name__ == "__main__":
    agent = ReasoningGraphAgent()
    test_plan = {
        'success': True,
        'data': {
            'retrievers': ['vector', 'bm25'],
            'top_k': 5,
            'rerank': True,
            'multi_round': True,
            'expand_keywords': ['香菇', '种植', '温度', '湿度', '种植技术', '环境控制'],
            'mode': 'complex',
            'sub_tasks': ['北京香菇种植适宜温度', '北京香菇种植适宜湿度']
        },
        'error': None
    }
    result = agent.run("北京种植香菇需要控制哪些温度和湿度条件？", test_plan)
    print(json.dumps({
        "success": result["success"],
        "called_tools": result["called_tools"],
        "optimized_answer": result["optimized_answer"],
        "error": result.get("error")
    }, ensure_ascii=False, indent=2))
