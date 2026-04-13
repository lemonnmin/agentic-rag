#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   evaluation.py
@Time    :   2026/03/15 23:00:00
@Author  :   lemonnmin
@Version :   1.2
@Desc    :   评估代理：彻底修复切片错误，增强类型安全
"""

import sys
import os
import re
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Optional, TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv


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


class EvaluationResult(BaseModel):
    retrieval_relevance: int = Field(ge=0, le=5, description="检索结果相关性（0-5）")
    answer_accuracy: int = Field(ge=0, le=5, description="答案准确性（0-5）")
    answer_completeness: int = Field(ge=0, le=5, description="答案完整性（0-5）")
    reasoning_effectiveness: int = Field(ge=0, le=5, description="推理有效性（0-5）")
    tool_call_appropriateness: int = Field(ge=0, le=5, description="工具调用合理性（0-5）")
    result_fusion_quality: int = Field(ge=0, le=5, description="结果融合质量（0-5）")
    answer_optimization_effect: int = Field(ge=0, le=5, description="答案优化效果（0-5）")
    suggestion: str = Field(description="优化建议")


# ========== 状态定义 ==========
class EvaluationState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    reasoning_result: Dict
    fused_results: List[Dict]
    raw_answer: str
    optimized_answer: str
    retrieve_rounds: int
    called_tools: List[str]
    tool_results: List[Dict]
    fused_results_str: str
    tool_results_str: str
    llm_response: Optional[str]
    evaluation_data: Optional[Dict]
    evaluation_result: Optional[Dict]
    error: Optional[str]
    step: str
    thread_id: Optional[str]


# ========== 全局OpenAI客户端 ==========
_client_instance = None

def get_openai_client():
    global _client_instance
    if _client_instance is None:
        load_env_from_root()
        _client_instance = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    return _client_instance


# ========== 评估提示词 ==========
EVAL_PROMPT_TEMPLATE = """
你是食用菌种植RAG系统评估专家，需评估以下维度（0-5分，5分为最优）：
【基础维度】
1. 检索结果相关性：检索内容（含RAG+工具）与问题的匹配程度
2. 答案准确性：答案是否符合食用菌种植常识（含工具返回数据的准确性）
3. 答案完整性：是否覆盖问题的所有核心需求（含实时/地区性需求）
4. 推理有效性：交织推理-检索循环是否有效

【系统已有条件】
1.工具类(RagSearchTool/TavilySearchTool/WeatherTool)
2.可供rag的食用菌种植知识库
3.交织推理-检索循环agent

【ReasoningAgent扩展维度】
5. 工具调用合理性：是否必要调用工具、工具选择是否正确、输入参数是否准确
6. 结果融合质量：多源结果的去重、优先级排序、整合效果
7. 答案优化效果：优化后答案相比原始答案的提升

评估素材：
- 用户问题：{query}
- 检索结果（融合后）：{fused_results}
- 原始答案：{raw_answer}
- 优化后答案：{optimized_answer}
- 推理轮数：{retrieve_rounds}
- 调用的工具：{called_tools}
- 工具调用结果：{tool_results}

输出要求：
1. 严格返回纯JSON格式，字段必须完全匹配EvaluationResult类定义
2. 建议需具体可落地
3. 分数需客观
4. 请严格按照以下格式返回：
{{
    "retrieval_relevance": 3,
    "answer_accuracy": 2,
    "answer_completeness": 4,
    "reasoning_effectiveness": 3,
    "tool_call_appropriateness": 5,
    "result_fusion_quality": 3,
    "answer_optimization_effect": 4,
    "suggestion": "改进建议..."
}}
"""

# ========== 节点函数 ==========
def parse_input_node(state: EvaluationState) -> dict:
    query = state.get("query", "")
    reasoning_result = state.get("reasoning_result", {})
    
    fused_results = reasoning_result.get("fused_results", [])
    raw_answer = reasoning_result.get("raw_answer", "")
    optimized_answer = reasoning_result.get("optimized_answer", "")
    retrieve_rounds = reasoning_result.get("retrieve_rounds", 0)
    called_tools = reasoning_result.get("called_tools", [])
    tool_results = reasoning_result.get("tool_results", [])
    
    fused_count = len(fused_results) if isinstance(fused_results, list) else 0
    print(f"📋 解析输入数据完成：问题={query}，工具={called_tools}，融合结果数={fused_count}")
    
    return {
        "query": query,
        "reasoning_result": reasoning_result,
        "fused_results": fused_results if isinstance(fused_results, list) else [],
        "raw_answer": raw_answer,
        "optimized_answer": optimized_answer,
        "retrieve_rounds": retrieve_rounds,
        "called_tools": called_tools if isinstance(called_tools, list) else [],
        "tool_results": tool_results if isinstance(tool_results, list) else [],
        "step": "input_parsed",
        "messages": [AIMessage(content=f"输入解析完成，共{fused_count}条融合结果")]
    }


def format_results_node(state: EvaluationState) -> dict:
    fused_results = state.get("fused_results", [])
    tool_results = state.get("tool_results", [])
    
    if not isinstance(fused_results, list):
        fused_results = []
    if not isinstance(tool_results, list):
        tool_results = []
    
    fused_count = len(fused_results)
    tool_count = len(tool_results)
    
    # 格式化融合结果
    fused_items = []
    for i, r in enumerate(fused_results):
        try:
            if isinstance(r, dict):
                content = r.get('content', '')
                source = str(r.get('source', 'unknown'))
                priority = str(r.get('priority', 99))
                fused_items.append(f"{i+1}. 内容：{content} | 来源：{source} | 优先级：{priority}")
            else:
                fused_items.append(f"{i+1}. 内容：{r}")
        except Exception as e:
            fused_items.append(f"{i+1}. 内容：<格式错误>")
    fused_results_str = "\n".join(fused_items) if fused_items else "无"
    
    # 格式化工具调用结果
    tool_items = []
    for idx, tool in enumerate(tool_results):
        try:
            if isinstance(tool, dict):
                if tool.get('error'):
                    tool_items.append(f"- 工具{idx+1}：错误 - {tool['error']}")
                else:
                    data = tool.get('data', [])
                    if data and isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict):
                            source = str(first.get('source', 'unknown'))
                            content = first.get('content', '')
                            tool_items.append(f"- 工具{idx+1}（{source}）：{content}")
                        else:
                            tool_items.append(f"- 工具{idx+1}：{first}")
                    else:
                        tool_items.append(f"- 工具{idx+1}：无数据")
            else:
                tool_items.append(f"- 工具{idx+1}：{tool}")
        except Exception as e:
            tool_items.append(f"- 工具{idx+1}：<格式错误>")
    tool_results_str = "\n".join(tool_items) if tool_items else "无"
    
    print(f"📝 格式化完成：融合结果{fused_count}条，工具结果{tool_count}条")
    
    return {
        "fused_results_str": fused_results_str,
        "tool_results_str": tool_results_str,
        "step": "formatted",
        "messages": [AIMessage(content="结果格式化完成")]
    }


def call_llm_evaluation_node(state: EvaluationState) -> dict:
    query = state.get("query", "")
    fused_results_str = state.get("fused_results_str", "")
    raw_answer = state.get("raw_answer", "")
    optimized_answer = state.get("optimized_answer", "")
    retrieve_rounds = state.get("retrieve_rounds", 0)
    called_tools = state.get("called_tools", [])
    tool_results_str = state.get("tool_results_str", "")
    
    try:
        # 将 called_tools 转为字符串
        called_tools_str = str(called_tools) if isinstance(called_tools, list) else "[]"
        
        prompt = EVAL_PROMPT_TEMPLATE.format(
            query=query,
            fused_results=fused_results_str,
            raw_answer=raw_answer,
            optimized_answer=optimized_answer,
            retrieve_rounds=retrieve_rounds,
            called_tools=called_tools_str,
            tool_results=tool_results_str
        )
        
        client = get_openai_client()
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        raw_response = response.choices[0].message.content
        print(f"🤖 LLM评估完成，响应长度：{len(raw_response)}")
        
        return {
            "llm_response": raw_response,
            "step": "llm_called",
            "messages": [AIMessage(content="LLM评估完成")]
        }
    except Exception as e:
        error_msg = f"LLM调用失败：{str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "llm_call_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }


def parse_json_node(state: EvaluationState) -> dict:
    raw_response = state.get("llm_response", "")
    if not raw_response:
        return {"error": "无LLM响应", "step": "parse_failed", "messages": [AIMessage(content="❌ 无LLM响应")]}
    
    try:
        data = json.loads(raw_response)
        required_fields = [
            "retrieval_relevance", "answer_accuracy", "answer_completeness",
            "reasoning_effectiveness", "tool_call_appropriateness",
            "result_fusion_quality", "answer_optimization_effect", "suggestion"
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            # 尝试嵌套结构
            if "scores" in data and isinstance(data["scores"], dict):
                data = {**data["scores"], "suggestion": data.get("suggestion", "")}
            elif "basic_dimensions" in data and isinstance(data["basic_dimensions"], dict):
                basic = data["basic_dimensions"]
                data = {
                    "retrieval_relevance": basic.get("检索相关性", 0),
                    "answer_accuracy": basic.get("答案准确性", 0),
                    "answer_completeness": basic.get("答案完整性", 0),
                    "reasoning_effectiveness": basic.get("推理有效性", 0),
                    "tool_call_appropriateness": basic.get("工具调用恰当性", 0),
                    "result_fusion_quality": basic.get("结果融合质量", 0),
                    "answer_optimization_effect": basic.get("答案优化效果", 0),
                    "suggestion": data.get("suggestion", "")
                }
            else:
                raise ValueError(f"缺少字段: {missing}")
        
        print(f"✅ JSON解析成功，字段：{list(data.keys())}")
        return {"evaluation_data": data, "step": "json_parsed", "messages": [AIMessage(content="JSON解析成功")]}
    except Exception as e:
        # 尝试清理后解析
        try:
            clean = raw_response.strip()
            if clean.startswith("```") and clean.endswith("```"):
                start = clean.find("{")
                end = clean.rfind("}") + 1
                if start >= 0 and end > start:
                    clean = clean[start:end]
            clean = re.sub(r'```\w*\n?', '', clean)
            clean = re.sub(r',\s*}', '}', clean)
            clean = re.sub(r',\s*]', ']', clean)
            clean = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', clean)
            data = json.loads(clean)
            print(f"✅ JSON解析成功（清理后）")
            return {"evaluation_data": data, "step": "json_parsed", "messages": [AIMessage(content="JSON解析成功（清理后）")]}
        except Exception as e2:
            error_msg = f"JSON解析失败：{str(e2)}"
            print(f"⚠️ {error_msg}\n原始响应：{raw_response[:200]}...")
            return {"error": error_msg, "step": "parse_failed", "messages": [AIMessage(content=f"⚠️ {error_msg}")]}


def validate_result_node(state: EvaluationState) -> dict:
    eval_data = state.get("evaluation_data", {})
    try:
        # 确保数字字段是整数
        for key in ["retrieval_relevance", "answer_accuracy", "answer_completeness",
                   "reasoning_effectiveness", "tool_call_appropriateness",
                   "result_fusion_quality", "answer_optimization_effect"]:
            if key in eval_data:
                try:
                    eval_data[key] = int(eval_data[key])
                except:
                    eval_data[key] = 0
        validated = EvaluationResult(**eval_data)
        print(f"✅ 结果验证通过，分数：{validated.retrieval_relevance}/{validated.answer_accuracy}/{validated.answer_completeness}")
        return {
            "evaluation_result": validated.model_dump(),
            "step": "validated",
            "messages": [AIMessage(content="✅ 评估结果验证通过")]
        }
    except Exception as e:
        error_msg = f"结果验证失败：{str(e)}"
        print(f"⚠️ {error_msg}")
        fallback = EvaluationResult(
            retrieval_relevance=0, answer_accuracy=0, answer_completeness=0,
            reasoning_effectiveness=0, tool_call_appropriateness=0,
            result_fusion_quality=0, answer_optimization_effect=0,
            suggestion=f"评估结果解析失败：{str(e)}，请检查LLM输出格式"
        )
        return {
            "evaluation_result": fallback.model_dump(),
            "error": error_msg,
            "step": "validated_with_fallback",
            "messages": [AIMessage(content="⚠️ 使用兜底评估结果")]
        }


def finalize_node(state: EvaluationState) -> dict:
    return {"step": "completed", "messages": [AIMessage(content="评估工作流完成")]}


def error_handler_node(state: EvaluationState) -> dict:
    error = state.get("error", "未知错误")
    fallback = EvaluationResult(
        retrieval_relevance=0, answer_accuracy=0, answer_completeness=0,
        reasoning_effectiveness=0, tool_call_appropriateness=0,
        result_fusion_quality=0, answer_optimization_effect=0,
        suggestion=f"评估过程出错：{error}"
    )
    return {
        "evaluation_result": fallback.model_dump(),
        "error": error,
        "step": "error_handled",
        "messages": [AIMessage(content=f"❌ 处理出错：{error}")]
    }


# ========== 条件边函数 ==========
def after_llm_call(state: EvaluationState) -> str:
    return "error_handler" if state.get("error") else "parse_json"

def after_parse(state: EvaluationState) -> str:
    return "error_handler" if state.get("error") else "validate_result"


# ========== 构建工作流图 ==========
def create_evaluation_agent():
    workflow = StateGraph(EvaluationState)
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("format_results", format_results_node)
    workflow.add_node("call_llm", call_llm_evaluation_node)
    workflow.add_node("parse_json", parse_json_node)
    workflow.add_node("validate_result", validate_result_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.add_edge(START, "parse_input")
    workflow.add_edge("parse_input", "format_results")
    workflow.add_edge("format_results", "call_llm")
    workflow.add_conditional_edges("call_llm", after_llm_call, {"parse_json": "parse_json", "error_handler": "error_handler"})
    workflow.add_conditional_edges("parse_json", after_parse, {"validate_result": "validate_result", "error_handler": "error_handler"})
    workflow.add_edge("validate_result", "finalize")
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)


# ========== 对外接口 ==========
class EvaluationGraphAgent:
    def __init__(self):
        self.app = create_evaluation_agent()

    def evaluate(self, query: str, reasoning_agent_result: Dict) -> Dict:
        thread_id = f"evaluation-{hash(query)}-{hash(str(reasoning_agent_result))}"
        inputs = {
            "messages": [HumanMessage(content=f"开始评估：{query[:50]}...")],
            "query": query,
            "reasoning_result": reasoning_agent_result,
            "step": "start"
        }
        try:
            result = self.app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
            return {
                "success": not result.get("error"),
                "data": result.get("evaluation_result", {}),
                "step": result.get("step", "completed"),
                "error": result.get("error")
            }
        except Exception as e:
            error_msg = f"工作流执行失败：{str(e)}"
            print(f"❌ {error_msg}")
            fallback = EvaluationResult(
                retrieval_relevance=0, answer_accuracy=0, answer_completeness=0,
                reasoning_effectiveness=0, tool_call_appropriateness=0,
                result_fusion_quality=0, answer_optimization_effect=0,
                suggestion=error_msg
            )
            return {"success": False, "data": fallback.model_dump(), "error": error_msg}


if __name__ == "__main__":
    agent = EvaluationGraphAgent()
    mock = {
        "success": True,
        "fused_results": [
            {"content": "香菇适宜温度15-25℃", "source": "vector", "priority": 0},
            {"content": "北京2025年3月平均温度10-18℃", "source": "weather", "priority": 2},
        ],
        "raw_answer": "香菇适宜15-25℃种植",
        "optimized_answer": "北京2025年3月种植香菇需控制温度在15-25℃...",
        "retrieve_rounds": 1,
        "called_tools": ["weather"],
        "tool_results": [
            {"success": True, "data": [{"content": "北京2025年3月平均温度10-18℃", "source": "weather"}]}
        ],
        "error": None
    }
    res = agent.evaluate("北京种植香菇温度湿度", mock)
    print(json.dumps(res, ensure_ascii=False, indent=2))