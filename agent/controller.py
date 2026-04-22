#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   controller_with_optimization.py
@Time    :   2026/03/15 23:45:00
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   增强版控制器：根据评估结果自动优化系统
"""

import builtins
import sys
import os
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Optional, TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from dotenv import load_dotenv

# 导入各个Agent
from agent.intent import parse_intent
from agent.planner import plan_retrieval
from agent.reasoning import ReasoningGraphAgent
from agent.evaluation import EvaluationGraphAgent
from rag.LLM import OpenAIChat
from rag.storage_db import get_storage_db


def safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sanitized_args = []
        for arg in args:
            text = str(arg)
            sanitized_args.append(text.encode(encoding, errors="ignore").decode(encoding, errors="ignore"))
        builtins.print(*sanitized_args, **kwargs)


print = safe_print


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


# ========== 状态定义 ==========
class ControllerState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str                              # 用户原始问题
    intent_result: Optional[Dict]            # 意图解析结果
    planner_result: Optional[Dict]           # 检索规划结果
    reasoning_result: Optional[Dict]         # 推理代理结果
    evaluation_result: Optional[Dict]        # 评估结果
    evaluation_context: Optional[Dict]       # 评测上下文
    optimization_suggestions: List[str]      # 优化建议列表
    optimization_applied: bool                # 是否已应用优化
    retry_count: int                          # 重试次数
    max_retries: int                          # 最大重试次数
    error: Optional[str]                      # 错误信息
    step: str                                 # 当前步骤
    thread_id: Optional[str]                  # 线程ID
    baseline_evaluation: Optional[Dict]       # 首次评估结果
    optimization_analysis: Optional[Dict]     # 结构化优化分析
    optimization_actions_applied: List[str]   # 已应用的优化动作
    out_of_scope: bool                        # 是否越界
    boundary_reason: Optional[str]            # 越界原因
    reasoning_degraded: bool                  # 推理是否降级
    evaluation_degraded: bool                 # 评估是否降级
    optimization_report: Optional[str]        # 优化报告


# ========== 优化规则引擎 ==========
NUMERIC_SCORE_KEYS = [
    "retrieval_relevance",
    "answer_accuracy",
    "answer_completeness",
    "reasoning_effectiveness",
    "tool_call_appropriateness",
    "result_fusion_quality",
    "answer_optimization_effect",
]

DOMAIN_KEYWORDS = [
    "食用菌", "香菇", "平菇", "金针菇", "白玉菇", "杏鲍菇", "菌丝", "菌种",
    "出菇", "发菌", "菌房", "培养料", "培养基", "灭菌", "接种", "栽培", "种植",
    "温度", "湿度", "通风", "污染", "菇", "菌",
]


def _is_out_of_scope_query(query: str, intent_data: Dict[str, Any]) -> bool:
    text = query or ""
    if any(keyword in text for keyword in DOMAIN_KEYWORDS):
        return False

    intent_type = intent_data.get("intent_type", "unknown")
    keywords = intent_data.get("keywords") or []
    domain = intent_data.get("domain", "")
    return (intent_type == "unknown" or not keywords) and domain in {"其他食用菌", "通用", "", None}


def _build_out_of_scope_result(reason: str) -> Dict[str, Any]:
    answer = (
        "该问题超出本系统当前的服务范围。"
        "本系统主要面向食用菌种植知识问答，可回答菌种选择、培养料配制、环境控制、出菇管理和病害防治等相关问题。"
        "请改为咨询食用菌种植相关内容。"
    )
    return {
        "reasoning_result": {
            "success": True,
            "fused_results": [],
            "raw_answer": answer,
            "optimized_answer": answer,
            "retrieve_rounds": 0,
            "called_tools": [],
            "tool_results": [],
            "error": None,
        },
        "evaluation_result": {
            "success": True,
            "data": {
                "evaluation_mode": "boundary",
                "retrieval_relevance": 0,
                "answer_accuracy": 4,
                "answer_completeness": 4,
                "reasoning_effectiveness": 3,
                "boundary_recognition": 5,
                "scope_compliance": 5,
                "response_clarity": 4,
                "helpful_redirection": 4,
                "tool_call_appropriateness": 5,
                "result_fusion_quality": 3,
                "answer_optimization_effect": 3,
                "final_score_basis": "Average the five boundary dimensions",
                "suggestion": "该问题超出系统服务范围，已触发边界拒答策略。",
            },
            "error": None,
        },
        "optimization_suggestions": [],
        "out_of_scope": True,
        "boundary_reason": reason,
    }


def _build_degraded_reasoning_result(error_msg: str) -> Dict[str, Any]:
    answer = (
        "系统在当前问题的深度推理阶段出现超时或异常，已返回降级结果。"
        "建议稍后重试，或将问题改写为更简洁、聚焦的食用菌种植问题。"
    )
    return {
        "success": False,
        "fused_results": [],
        "raw_answer": answer,
        "optimized_answer": answer,
        "retrieve_rounds": 0,
        "called_tools": [],
        "tool_results": [],
        "error": error_msg,
    }


def _build_degraded_evaluation_result(error_msg: str) -> Dict[str, Any]:
    return {
        "success": False,
        "data": {
            "evaluation_mode": "standard",
            "retrieval_relevance": 2,
            "answer_accuracy": 2,
            "answer_completeness": 2,
            "reasoning_effectiveness": 2,
            "tool_call_appropriateness": 2,
            "result_fusion_quality": 2,
            "answer_optimization_effect": 2,
            "suggestion": f"评估阶段降级：{error_msg}",
        },
        "error": error_msg,
    }


def _compute_final_score(evaluation_data: Dict[str, Any]) -> float:
    if not evaluation_data:
        return 0.0

    evaluation_mode = evaluation_data.get("evaluation_mode", "standard")
    if evaluation_mode == "boundary":
        boundary_keys = [
            "boundary_recognition",
            "scope_compliance",
            "response_clarity",
            "helpful_redirection",
            "tool_call_appropriateness",
        ]
        scores = [float(evaluation_data.get(key, 0) or 0) for key in boundary_keys]
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    standard_keys = [
        "retrieval_relevance",
        "answer_accuracy",
        "answer_completeness",
    ]
    scores = [float(evaluation_data.get(key, 0) or 0) for key in standard_keys]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


class OptimizationEngine:
    """根据评估结果自动生成优化策略"""
    
    def __init__(self):
        self.llm = OpenAIChat()
        
    def analyze_evaluation(self, evaluation: Dict) -> Dict:
        """分析评估结果，生成优化建议"""
        data = evaluation.get("data", {})
        
        # 提取各维度分数
        scores = {
            "retrieval_relevance": data.get("retrieval_relevance", 0),
            "answer_accuracy": data.get("answer_accuracy", 0),
            "answer_completeness": data.get("answer_completeness", 0),
            "reasoning_effectiveness": data.get("reasoning_effectiveness", 0),
            "tool_call_appropriateness": data.get("tool_call_appropriateness", 0),
            "result_fusion_quality": data.get("result_fusion_quality", 0),
            "answer_optimization_effect": data.get("answer_optimization_effect", 0)
        }
        
        suggestion = data.get("suggestion", "")
        
        # 识别低分维度（<3分）
        low_score_dimensions = [dim for dim, score in scores.items() if score < 3]
        
        # 提取具体优化建议
        optimization_actions = self._extract_actions(suggestion)
        priority_issue = min(scores, key=scores.get) if scores else None
        recommended_actions = self._build_recommended_actions(low_score_dimensions)
        
        return {
            "scores": scores,
            "low_score_dimensions": low_score_dimensions,
            "priority_issue": priority_issue,
            "original_suggestion": suggestion,
            "optimization_actions": optimization_actions,
            "recommended_actions": recommended_actions
        }
    
    def _extract_actions(self, suggestion: str) -> List[Dict]:
        """从建议文本中提取可执行的操作"""
        actions = []
        
        # 规则：提取带编号的建议
        lines = suggestion.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or '建议' in line or '改进' in line):
                # 识别操作类型
                action_type = "unknown"
                if '检索' in line or 'rag' in line.lower():
                    action_type = "retrieval"
                elif '推理' in line or 'reasoning' in line.lower():
                    action_type = "reasoning"
                elif '工具' in line or 'tool' in line.lower():
                    action_type = "tool"
                elif '融合' in line or 'fusion' in line.lower():
                    action_type = "fusion"
                elif '答案' in line or 'answer' in line.lower():
                    action_type = "answer"
                
                actions.append({
                    "type": action_type,
                    "description": line,
                    "applied": False
                })
        
        return actions

    def _build_recommended_actions(self, low_score_dimensions: List[str]) -> List[str]:
        actions: List[str] = []

        if "retrieval_relevance" in low_score_dimensions:
            actions.extend(["increase_top_k", "enable_rerank", "enable_multi_round", "expand_keywords"])

        if "answer_completeness" in low_score_dimensions:
            actions.extend(["enable_multi_round", "expand_keywords", "structured_answer"])

        if "reasoning_effectiveness" in low_score_dimensions:
            actions.extend(["enable_multi_round", "increase_max_rounds", "focus_subtasks"])

        if "tool_call_appropriateness" in low_score_dimensions:
            actions.extend(["force_tool_call", "prefer_suggested_tools"])

        if "result_fusion_quality" in low_score_dimensions:
            actions.extend(["enable_rerank", "prefer_high_quality_context"])

        if "answer_optimization_effect" in low_score_dimensions:
            actions.extend(["detailed_optimization", "structured_answer", "prefer_high_quality_context"])

        return list(dict.fromkeys(actions))
    
    def generate_improved_plan(self, original_plan: Dict, evaluation_analysis: Dict) -> Dict:
        """根据评估分析生成改进的检索计划"""
        
        low_score_dims = evaluation_analysis.get("low_score_dimensions", [])
        actions = evaluation_analysis.get("optimization_actions", [])
        recommended_actions = evaluation_analysis.get("recommended_actions", [])
        
        # 复制原始计划
        improved_plan = original_plan.copy() if original_plan else {"data": {}}
        
        # 确保有data字段
        if "data" not in improved_plan:
            improved_plan["data"] = {}
        
        plan_data = improved_plan["data"]
        plan_data["optimization_actions"] = recommended_actions
        
        # 根据低分维度自动调整参数
        if "retrieval_relevance" in low_score_dims:
            # 检索相关性低：增加top_k，启用多轮检索
            plan_data["top_k"] = min(plan_data.get("top_k", 5) + 2, 10)
            plan_data["multi_round"] = True
            plan_data["rerank"] = True
            print(f"📊 优化：检索相关性低，top_k调整为{plan_data['top_k']}，启用多轮检索")
        
        if "answer_completeness" in low_score_dims:
            # 答案完整性低：增加检索轮次，扩展关键词
            plan_data["multi_round"] = True
            plan_data["structured_answer"] = True
            if "expand_keywords" not in plan_data:
                plan_data["expand_keywords"] = []
            plan_data["expand_keywords"].extend(["详细", "完整", "全部"])
            print(f"📊 优化：答案完整性低，启用多轮检索，添加扩展关键词")

        if "reasoning_effectiveness" in low_score_dims:
            plan_data["multi_round"] = True
            plan_data["max_rounds"] = max(plan_data.get("max_rounds", 2), 3)
            plan_data["focus_subtasks"] = True
            print("📊 优化：推理有效性低，增加检索轮次并聚焦子任务")
        
        if "tool_call_appropriateness" in low_score_dims:
            # 工具调用问题：强制在下次尝试中调用工具
            plan_data["force_tool_call"] = True
            plan_data["prefer_suggested_tools"] = True
            print(f"📊 优化：工具调用问题，下次将强制评估工具需求")
        
        if "result_fusion_quality" in low_score_dims:
            # 结果融合质量低：增加rerank
            plan_data["rerank"] = True
            plan_data["prefer_high_quality_context"] = True
            print(f"📊 优化：结果融合质量低，启用重排序")
        
        if "answer_optimization_effect" in low_score_dims:
            # 答案优化效果低：在优化阶段提供更详细的上下文
            plan_data["detailed_optimization"] = True
            plan_data["structured_answer"] = True
            plan_data["prefer_high_quality_context"] = True
            print(f"📊 优化：答案优化效果低，将提供更详细的上下文")
        
        # 根据具体建议调整
        for action in actions:
            if "城市" in action["description"] and "天气" in action["description"]:
                # 建议调用天气工具
                plan_data["suggested_tools"] = plan_data.get("suggested_tools", [])
                if "weather" not in plan_data["suggested_tools"]:
                    plan_data["suggested_tools"].append("weather")
                print(f"📊 优化：根据建议添加天气工具调用")
            
            if "多轮" in action["description"] or "交织" in action["description"]:
                plan_data["multi_round"] = True
                plan_data["max_rounds"] = 3
                print(f"📊 优化：根据建议启用多轮交织检索")

        if "expand_keywords" in plan_data:
            plan_data["expand_keywords"] = list(dict.fromkeys(plan_data["expand_keywords"]))
        
        return improved_plan
    
    def generate_optimization_report(self, before: Dict, after: Dict, analysis: Dict = None) -> str:
        """生成优化前后对比报告"""
        analysis = analysis or {}
        report = f"""
## 优化效果报告

### 优化前评估分数
- 检索相关性: {before.get('retrieval_relevance', 0)}/5
- 答案准确性: {before.get('answer_accuracy', 0)}/5
- 答案完整性: {before.get('answer_completeness', 0)}/5
- 推理有效性: {before.get('reasoning_effectiveness', 0)}/5
- 工具调用合理性: {before.get('tool_call_appropriateness', 0)}/5
- 结果融合质量: {before.get('result_fusion_quality', 0)}/5
- 答案优化效果: {before.get('answer_optimization_effect', 0)}/5

### 优化后评估分数
- 检索相关性: {after.get('retrieval_relevance', 0)}/5
- 答案准确性: {after.get('answer_accuracy', 0)}/5
- 答案完整性: {after.get('answer_completeness', 0)}/5
- 推理有效性: {after.get('reasoning_effectiveness', 0)}/5
- 工具调用合理性: {after.get('tool_call_appropriateness', 0)}/5
- 结果融合质量: {after.get('result_fusion_quality', 0)}/5
- 答案优化效果: {after.get('answer_optimization_effect', 0)}/5

### 改进幅度
"""
        priority_issue = analysis.get("priority_issue")
        applied_actions = analysis.get("recommended_actions", [])
        if priority_issue:
            report += f"- 本次优化重点: {priority_issue}\n"
        if applied_actions:
            report += f"- 应用动作: {', '.join(applied_actions)}\n"
        # 计算改进幅度
        for key in before.keys():
            if key in after:
                diff = after[key] - before[key]
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                report += f"- {key}: {diff:+d} {arrow}\n"
        
        return report


# ========== 增强节点函数 ==========
def intent_node(state: ControllerState) -> dict:
    """节点1：执行意图理解"""
    query = state["query"]
    
    try:
        print(f"\n🔍 [步骤1] 意图理解：{query}")
        intent_result = parse_intent(query)
        
        if intent_result.get("success"):
            print(f"✅ 意图理解成功：类型={intent_result['data'].get('intent_type')}, 复杂度={intent_result['data'].get('complexity')}")
        else:
            print(f"⚠️ 意图理解降级使用规则兜底")
        
        return {
            "intent_result": intent_result,
            "step": "intent_completed",
            "messages": [AIMessage(content=f"意图理解完成：{intent_result['data'].get('intent_type')}")]
        }
    except Exception as e:
        error_msg = f"意图理解失败：{str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "intent_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }


def planner_node(state: ControllerState) -> dict:
    """节点2：执行检索规划（支持优化）"""
    if state.get("out_of_scope"):
        print("\n[Step 2] Out-of-scope query, skip planner.")
        return {
            "step": "planner_skipped",
            "messages": [AIMessage(content="Skip planner for out-of-scope query.")],
        }

    intent_result = state["intent_result"]
    optimization_suggestions = state.get("optimization_suggestions", [])
    
    try:
        print(f"\n📋 [步骤2] 检索规划")
        
        # 执行初始规划
        planner_result = plan_retrieval(intent_result)
        
        # 如果有优化建议，根据建议调整规划
        if optimization_suggestions and state.get("retry_count", 0) > 0:
            print(f"🔄 根据评估建议调整检索规划...")
            
            # 根据优化建议调整参数
            if "检索相关性" in str(optimization_suggestions):
                if "data" in planner_result:
                    planner_result["data"]["top_k"] = min(planner_result["data"].get("top_k", 5) + 2, 10)
                    planner_result["data"]["multi_round"] = True
                    print(f"  - 调整：增加top_k到{planner_result['data']['top_k']}，启用多轮检索")
            
            if "工具调用" in str(optimization_suggestions):
                if "data" in planner_result:
                    planner_result["data"]["force_tool_evaluation"] = True
                    print(f"  - 调整：强制评估工具调用需求")
        
        if planner_result.get("success"):
            print(f"✅ 检索规划成功：模式={planner_result['data'].get('mode')}, top_k={planner_result['data'].get('top_k')}")
        else:
            print(f"⚠️ 检索规划降级使用默认策略")
        
        return {
            "planner_result": planner_result,
            "step": "planner_completed",
            "messages": [AIMessage(content=f"检索规划完成：模式={planner_result['data'].get('mode')}")]
        }
    except Exception as e:
        error_msg = f"检索规划失败：{str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "planner_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }


import concurrent.futures
import time

def reasoning_node(state: ControllerState) -> dict:
    query = state["query"]
    planner_result = state["planner_result"]
    retry_count = state.get("retry_count", 0)
    
    print(f"\n🤔 [步骤3] 推理代理执行中... (尝试 #{retry_count + 1}) 开始时间：{time.strftime('%H:%M:%S')}")
    
    # 创建代理实例（每次调用新实例，但内部可能有缓存，问题不大）
    reasoning_agent = ReasoningGraphAgent()
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(reasoning_agent.run, query, planner_result)
    
    try:
        # 设置超时 60 秒（可根据需要调整）
        reasoning_result = future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        error_msg = f"推理代理执行超时（60秒），尝试 #{retry_count + 1} 失败"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "reasoning_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }
    except Exception as e:
        executor.shutdown(wait=False)
        error_msg = f"推理代理执行异常：{str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "reasoning_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }
    finally:
        executor.shutdown(wait=False)
    
    # 正常处理结果
    if reasoning_result.get("success"):
        print(f"✅ 推理代理成功：检索轮数={reasoning_result.get('retrieve_rounds')}, 调用工具={reasoning_result.get('called_tools')}")
        print(f"📝 优化答案预览：{reasoning_result.get('optimized_answer', '')[:100]}...")
    else:
        print(f"⚠️ 推理代理执行失败，使用兜底")
    
    return {
        "reasoning_result": reasoning_result,
        "step": "reasoning_completed",
        "messages": [AIMessage(content="推理代理执行完成")]
    }


def evaluation_node(state: ControllerState) -> dict:
    query = state["query"]
    reasoning_result = state["reasoning_result"]
    
    print(f"\n📊 [步骤4] 评估代理执行中... 开始时间：{time.strftime('%H:%M:%S')}")
    
    evaluation_agent = EvaluationGraphAgent()
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        evaluation_agent.evaluate,
        query,
        reasoning_result,
        state.get("evaluation_context") or {},
    )
    
    try:
        # 设置超时 30 秒
        evaluation_result = future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        error_msg = "评估代理执行超时（30秒）"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "evaluation_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }
    except Exception as e:
        executor.shutdown(wait=False)
        error_msg = f"评估代理执行异常：{str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "step": "evaluation_failed",
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }
    finally:
        executor.shutdown(wait=False)
    
    if evaluation_result.get("success"):
        data = evaluation_result.get("data", {})
        print(f"✅ 评估成功：相关性={data.get('retrieval_relevance')}, 准确性={data.get('answer_accuracy')}, 完整性={data.get('answer_completeness')}")
        
        # 分析评估结果，提取优化建议
        optimization_engine = OptimizationEngine()
        analysis = optimization_engine.analyze_evaluation(evaluation_result)
        suggestions = [a["description"] for a in analysis.get("optimization_actions", [])]
        print(f"💡 生成{len(suggestions)}条优化建议")
        
        return {
            "evaluation_result": evaluation_result,
            "optimization_suggestions": suggestions,
            "optimization_analysis": analysis,
            "step": "evaluation_completed",
            "messages": [AIMessage(content="评估完成，已生成优化建议")]
        }
    else:
        print(f"⚠️ 评估代理执行失败")
        return {
            "evaluation_result": evaluation_result,
            "optimization_suggestions": [],
            "step": "evaluation_completed",
            "messages": [AIMessage(content="评估完成，但结果可能不完整")]
        }


def optimization_node(state: ControllerState) -> dict:
    """节点5：根据评估结果决定是否重新执行"""
    evaluation_result = state.get("evaluation_result", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    data = evaluation_result.get("data", {})
    
    # 计算7维平均分，并针对关键维度设置更严格的兜底阈值
    scores = [
        data.get("retrieval_relevance", 0),
        data.get("answer_accuracy", 0),
        data.get("answer_completeness", 0),
        data.get("reasoning_effectiveness", 0),
        data.get("tool_call_appropriateness", 0),
        data.get("result_fusion_quality", 0),
        data.get("answer_optimization_effect", 0)
    ]
    avg_score = sum(scores) / len(scores)
    critical_dimensions = {
        "answer_accuracy": data.get("answer_accuracy", 0),
        "answer_completeness": data.get("answer_completeness", 0),
        "retrieval_relevance": data.get("retrieval_relevance", 0),
        "tool_call_appropriateness": data.get("tool_call_appropriateness", 0),
    }
    has_critical_issue = any(score < 3 for score in critical_dimensions.values())
    
    # 决策逻辑：平均分偏低或关键维度过低，且还有重试次数，则进入优化重试
    if (avg_score < 3.5 or has_critical_issue) and retry_count < max_retries:
        print(f"\n🔄 [步骤5] 评估分数较低 ({avg_score:.1f}/5)，启动优化重试 (第{retry_count + 1}次)")
        
        # 根据评估结果调整规划
        optimization_engine = OptimizationEngine()
        analysis = state.get("optimization_analysis") or optimization_engine.analyze_evaluation(evaluation_result)
        
        # 生成改进的规划
        original_plan = state.get("planner_result", {})
        improved_plan = optimization_engine.generate_improved_plan(original_plan, analysis)
        
        return {
            "planner_result": improved_plan,
            "retry_count": retry_count + 1,
            "optimization_applied": True,
            "baseline_evaluation": state.get("baseline_evaluation") or evaluation_result,
            "optimization_analysis": analysis,
            "optimization_actions_applied": analysis.get("recommended_actions", []),
            "step": "optimization_applied",
            "messages": [AIMessage(content=f"应用优化策略，准备重试 (第{retry_count + 1}次)")]
        }
    else:
        if retry_count >= max_retries:
            print(f"\n✅ [步骤5] 已达最大重试次数 ({max_retries})，结束流程")
        else:
            print(f"\n✅ [步骤5] 评估分数达标 ({avg_score:.1f}/5)，结束流程")
        
        return {
            "optimization_applied": False,
            "step": "optimization_skipped",
            "messages": [AIMessage(content="评估达标，无需优化重试")]
        }


def finalize_node(state: ControllerState) -> dict:
    """节点6：整理最终结果"""
    print(f"\n✨ [步骤6] 整理最终结果")
    
    # 生成优化报告（如果进行了优化）
    optimization_report = None
    if state.get("retry_count", 0) > 0:
        optimization_engine = OptimizationEngine()
        before_eval = (state.get("baseline_evaluation") or {}).get("data", {})
        after_eval = (state.get("evaluation_result") or {}).get("data", {})
        if before_eval and after_eval:
            optimization_report = optimization_engine.generate_optimization_report(
                before_eval,
                after_eval,
                state.get("optimization_analysis") or {}
            )
        else:
            optimization_report = "已根据评估建议进行优化重试"
    
    return {
        "step": "completed",
        "optimization_report": optimization_report,
        "messages": [AIMessage(content="控制器工作流完成")]
    }


def error_handler_node(state: ControllerState) -> dict:
    """错误处理节点"""
    error = state.get("error", "未知错误")
    current_step = state.get("step", "unknown")
    
    print(f"\n❌ 工作流执行出错（步骤：{current_step}）：{error}")
    
    return {
        "error": error,
        "step": "error_handled",
        "messages": [AIMessage(content=f"❌ 处理出错：{error}")]
    }


# ========== 条件边函数 ==========
def after_intent(state: ControllerState) -> str:
    """意图解析后的路由"""
    if state.get("error"):
        return "error_handler"
    return "planner"


def after_planner(state: ControllerState) -> str:
    """检索规划后的路由"""
    if state.get("error"):
        return "error_handler"
    return "reasoning"


def after_reasoning(state: ControllerState) -> str:
    """推理代理后的路由"""
    if state.get("error"):
        return "error_handler"
    return "evaluation"


def after_evaluation(state: ControllerState) -> str:
    """评估后的路由"""
    if state.get("error"):
        return "error_handler"
    return "optimization"


def after_optimization(state: ControllerState) -> str:
    """优化决策后的路由"""
    if state.get("error"):
        return "error_handler"
    
    # 如果应用了优化，重新执行推理
    if state.get("optimization_applied", False):
        return "reasoning"
    else:
        return "finalize"


# ========== 构建工作流图 ==========
def create_controller_agent():
    workflow = StateGraph(ControllerState)

    # 添加节点
    workflow.add_node("intent", intent_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("optimization", optimization_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)

    # 设置边
    workflow.add_edge(START, "intent")

    # 意图后路由
    workflow.add_conditional_edges(
        "intent",
        after_intent,
        {
            "planner": "planner",
            "error_handler": "error_handler"
        }
    )

    # 规划后路由
    workflow.add_conditional_edges(
        "planner",
        after_planner,
        {
            "reasoning": "reasoning",
            "error_handler": "error_handler"
        }
    )

    # 推理后路由
    workflow.add_conditional_edges(
        "reasoning",
        after_reasoning,
        {
            "evaluation": "evaluation",
            "error_handler": "error_handler"
        }
    )

    # 评估后路由
    workflow.add_conditional_edges(
        "evaluation",
        after_evaluation,
        {
            "optimization": "optimization",
            "error_handler": "error_handler"
        }
    )

    # 优化决策后路由
    workflow.add_conditional_edges(
        "optimization",
        after_optimization,
        {
            "reasoning": "reasoning",
            "finalize": "finalize",
            "error_handler": "error_handler"
        }
    )

    # 结束
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# ========== 对外接口 ==========
class OptimizedRAGController:
    """增强版RAG控制器：支持评估驱动的自动优化"""
    
    def __init__(self):
        self.app = create_controller_agent()
        self.optimization_engine = OptimizationEngine()
    
    def run(
        self,
        query: str,
        max_retries: int = 2,
        collection_name: Optional[str] = None,
        evaluation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        执行完整RAG流程，支持自动优化
        
        Args:
            query: 用户问题
            max_retries: 最大重试次数
            
        Returns:
            Dict: 包含意图、规划、推理结果、评估和优化报告的完整结果
        """
        thread_id = f"controller-optimized-{hash(query)}"
        
        inputs = {
            "messages": [HumanMessage(content=f"开始处理：{query}")],
            "query": query,
            "evaluation_context": evaluation_context or {},
            "retry_count": 0,
            "max_retries": max_retries,
            "optimization_suggestions": [],
            "optimization_actions_applied": [],
            "step": "start"
        }

        try:
            result = self.app.invoke(
                inputs,
                config={"configurable": {"thread_id": thread_id}}
            )

            # 构建完整返回结果
            intent_result = result.get("intent_result", {})
            planner_result = result.get("planner_result", {})
            reasoning_result = result.get("reasoning_result", {})
            evaluation_result = result.get("evaluation_result", {})
            
            # 计算最终分数
            eval_data = evaluation_result.get("data", {}) if evaluation_result else {}
            final_score = _compute_final_score(eval_data)

            payload = {
                "success": not result.get("error"),
                "query": query,
                "retry_count": result.get("retry_count", 0),
                "final_score": round(final_score, 2),
                "intent": intent_result.get("data", {}) if intent_result.get("success") else intent_result,
                "planner": planner_result.get("data", {}) if planner_result.get("success") else planner_result,
                "reasoning": {
                    "fused_results": reasoning_result.get("fused_results", []),
                    "raw_answer": reasoning_result.get("raw_answer", ""),
                    "optimized_answer": reasoning_result.get("optimized_answer", ""),
                    "retrieve_rounds": reasoning_result.get("retrieve_rounds", 0),
                    "called_tools": reasoning_result.get("called_tools", [])
                } if reasoning_result.get("success") else reasoning_result,
                "evaluation": eval_data,
                "optimization_suggestions": result.get("optimization_suggestions", []),
                "optimization_report": result.get("optimization_report"),
                "out_of_scope": result.get("out_of_scope", False),
                "boundary_reason": result.get("boundary_reason"),
                "domain_scope": (
                    intent_result.get("data", {}).get("domain_scope")
                    if isinstance(intent_result, dict) and intent_result.get("success")
                    else None
                ),
                "step": result.get("step", "completed"),
                "error": result.get("error")
            }
            try:
                logged_collection_name = collection_name
                planner_payload = payload.get("planner", {})
                if logged_collection_name is None and isinstance(planner_payload, dict):
                    logged_collection_name = planner_payload.get("collection_name")
                get_storage_db().log_query_result(payload, collection_name=logged_collection_name)
            except Exception as db_error:
                print(f"⚠️ SQLite日志写入失败：{db_error}")

            return payload
            
        except Exception as e:
            error_msg = f"控制器工作流执行失败：{str(e)}"
            print(f"❌ {error_msg}")
            
            payload = {
                "success": False,
                "query": query,
                "error": error_msg
            }
            try:
                get_storage_db().log_query_result(payload)
            except Exception as db_error:
                print(f"⚠️ SQLite日志写入失败：{db_error}")
            return payload


def print_optimized_result(result: Dict):
    """美化打印优化后的结果"""
    print("\n" + "="*80)
    print("🎯 最终结果 (优化版)")
    print("="*80)
    
    if not result.get("success"):
        print(f"❌ 执行失败：{result.get('error')}")
        return
    
    print(f"\n📊 最终评分：{result.get('final_score', 0)}/5")
    print(f"🔄 重试次数：{result.get('retry_count', 0)}")
    
    # 意图结果
    intent = result.get("intent", {})
    print(f"\n📌 意图理解：")
    print(f"  - 类型：{intent.get('intent_type')}")
    print(f"  - 领域：{intent.get('domain')}")
    print(f"  - 复杂度：{intent.get('complexity')}")
    
    # 规划结果
    planner = result.get("planner", {})
    print(f"\n📋 检索规划：")
    print(f"  - 模式：{planner.get('mode')}")
    print(f"  - top_k：{planner.get('top_k')}")
    
    # 推理结果
    reasoning = result.get("reasoning", {})
    print(f"\n🤔 推理结果：")
    print(f"  - 检索轮数：{reasoning.get('retrieve_rounds')}")
    print(f"  - 调用工具：{reasoning.get('called_tools')}")
    print(f"\n  ✨ 优化答案：")
    print(f"    {reasoning.get('optimized_answer', '')}")
    
    # 评估结果
    evaluation = result.get("evaluation", {})
    if evaluation:
        print(f"\n📊 评估结果：")
        print(f"  - 检索相关性：{evaluation.get('retrieval_relevance')}/5")
        print(f"  - 答案准确性：{evaluation.get('answer_accuracy')}/5")
        print(f"  - 答案完整性：{evaluation.get('answer_completeness')}/5")
        print(f"  - 推理有效性：{evaluation.get('reasoning_effectiveness')}/5")
        print(f"  - 工具调用合理性：{evaluation.get('tool_call_appropriateness')}/5")
        print(f"  - 结果融合质量：{evaluation.get('result_fusion_quality')}/5")
        print(f"  - 答案优化效果：{evaluation.get('answer_optimization_effect')}/5")
    
    # 优化建议
    suggestions = result.get("optimization_suggestions", [])
    if suggestions:
        print(f"\n💡 优化建议：")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion[:100]}...")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # 测试优化版控制器
    controller = OptimizedRAGController()
    
    # 测试问题
    test_queries = "山东临沂种植白玉菇该怎么在育种阶段需要注意什么，有什么步骤？"
    result = controller.run(test_queries, max_retries=2)
    print_optimized_result(result)
 
# ===== Runtime patches for stability =====
NUMERIC_SCORE_KEYS = [
    "retrieval_relevance",
    "answer_accuracy",
    "answer_completeness",
    "reasoning_effectiveness",
    "tool_call_appropriateness",
    "result_fusion_quality",
    "answer_optimization_effect",
]

DOMAIN_KEYWORDS = [
    "食用菌", "香菇", "平菇", "金针菇", "白玉菇", "杏鲍菇", "菌丝", "菌种",
    "出菇", "发菌", "菌房", "培养料", "培养基", "灭菌", "接种", "栽培", "种植",
    "温度", "湿度", "通风", "污染", "菇", "菌",
]


def _is_out_of_scope_query(query: str, intent_data: Dict[str, Any]) -> bool:
    text = query or ""
    if any(keyword in text for keyword in DOMAIN_KEYWORDS):
        return False
    intent_type = intent_data.get("intent_type", "unknown")
    keywords = intent_data.get("keywords") or []
    domain = intent_data.get("domain", "")
    return (intent_type == "unknown" or not keywords) and domain in {"其他食用菌", "通用", "", None}


def _build_out_of_scope_result(reason: str) -> Dict[str, Any]:
    answer = (
        "该问题超出本系统当前的服务范围。"
        "本系统主要面向食用菌种植知识问答，可回答菌种选择、培养料配制、环境控制、出菇管理和病害防治等相关问题。"
        "请改为咨询食用菌种植相关内容。"
    )
    return {
        "reasoning_result": {
            "success": True,
            "fused_results": [],
            "raw_answer": answer,
            "optimized_answer": answer,
            "retrieve_rounds": 0,
            "called_tools": [],
            "tool_results": [],
            "error": None,
        },
        "evaluation_result": {
            "success": True,
            "data": {
                "evaluation_mode": "boundary",
                "retrieval_relevance": 0,
                "answer_accuracy": 4,
                "answer_completeness": 4,
                "reasoning_effectiveness": 3,
                "boundary_recognition": 5,
                "scope_compliance": 5,
                "response_clarity": 4,
                "helpful_redirection": 4,
                "tool_call_appropriateness": 5,
                "result_fusion_quality": 3,
                "answer_optimization_effect": 3,
                "final_score_basis": "Average the five boundary dimensions",
                "suggestion": "该问题超出系统服务范围，已触发边界拒答策略。",
            },
            "error": None,
        },
        "optimization_suggestions": [],
        "out_of_scope": True,
        "boundary_reason": reason,
    }


def _build_degraded_reasoning_result(error_msg: str) -> Dict[str, Any]:
    answer = (
        "系统在当前问题的深度推理阶段出现超时或异常，已返回降级结果。"
        "建议稍后重试，或将问题改写为更简洁、聚焦的食用菌种植问题。"
    )
    return {
        "success": False,
        "fused_results": [],
        "raw_answer": answer,
        "optimized_answer": answer,
        "retrieve_rounds": 0,
        "called_tools": [],
        "tool_results": [],
        "error": error_msg,
    }


def _build_degraded_evaluation_result(error_msg: str) -> Dict[str, Any]:
    return {
        "success": False,
        "data": {
            "retrieval_relevance": 2,
            "answer_accuracy": 2,
            "answer_completeness": 2,
            "reasoning_effectiveness": 2,
            "tool_call_appropriateness": 2,
            "result_fusion_quality": 2,
            "answer_optimization_effect": 2,
            "suggestion": f"评估阶段降级：{error_msg}",
        },
        "error": error_msg,
    }


def _patched_generate_optimization_report(self, before: Dict, after: Dict, analysis: Dict = None) -> str:
    analysis = analysis or {}
    report = "## 优化效果报告\n\n"
    priority_issue = analysis.get("priority_issue")
    applied_actions = analysis.get("recommended_actions", [])
    if priority_issue:
        report += f"- 本次优化重点: {priority_issue}\n"
    if applied_actions:
        report += f"- 应用动作: {', '.join(applied_actions)}\n"
    for key in NUMERIC_SCORE_KEYS:
        before_value = before.get(key, 0)
        after_value = after.get(key, 0)
        if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
            continue
        diff = after_value - before_value
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        report += f"- {key}: {before_value} -> {after_value} ({diff:+.0f}) {arrow}\n"
    return report


OptimizationEngine.generate_optimization_report = _patched_generate_optimization_report


PATCHED_DOMAIN_KEYWORDS = [
    "食用菌",
    "香菇",
    "平菇",
    "金针菇",
    "白玉菇",
    "杏鲍菇",
    "菌丝",
    "菌种",
    "出菇",
    "发菌",
    "菌房",
    "培养料",
    "培养基",
    "灭菌",
    "接种",
    "栽培",
    "种植",
    "温度",
    "湿度",
    "通风",
    "污染",
]


def _patched_is_out_of_scope_query(query: str, intent_data: Dict[str, Any]) -> bool:
    domain_scope = (intent_data or {}).get("domain_scope")
    if domain_scope in {"in_domain", "mixed"}:
        return False
    if domain_scope == "out_of_scope":
        return True

    text = query or ""
    if any(keyword in text for keyword in PATCHED_DOMAIN_KEYWORDS):
        return False

    intent_type = (intent_data or {}).get("intent_type", "unknown")
    keywords = (intent_data or {}).get("keywords") or []
    domain = (intent_data or {}).get("domain", "")

    return (intent_type == "unknown" or not keywords) and domain in {"其他食用菌", "通用", "", None}


def intent_node(state: ControllerState) -> dict:
    """Patched intent node with out-of-scope early stop."""
    query = state["query"]

    try:
        print(f"\n[Step 1] Intent parsing: {query}")
        intent_result = parse_intent(query)
        intent_data = intent_result.get("data", {}) if intent_result.get("success") else {}

        if _patched_is_out_of_scope_query(query, intent_data):
            print("[Boundary] Out-of-scope query detected. Return boundary response directly.")
            boundary_result = _build_out_of_scope_result("out_of_scope_query")
            return {
                "intent_result": intent_result,
                **boundary_result,
                "step": "intent_completed",
                "messages": [AIMessage(content="Detected out-of-scope query and returned boundary response.")],
            }

        return {
            "intent_result": intent_result,
            "step": "intent_completed",
            "messages": [AIMessage(content=f"Intent parsed: {intent_data.get('intent_type', 'unknown')}")],
        }
    except Exception as e:
        error_msg = f"Intent parsing failed: {str(e)}"
        print(f"[Error] {error_msg}")
        return {
            "error": error_msg,
            "step": "intent_failed",
            "messages": [AIMessage(content=error_msg)],
        }


def reasoning_node(state: ControllerState) -> dict:
    """Patched reasoning node with degraded fallback instead of hard failure."""
    if state.get("out_of_scope"):
        print("\n[Step 3] Out-of-scope query, skip reasoning.")
        return {
            "reasoning_result": state.get("reasoning_result", {}),
            "reasoning_degraded": False,
            "step": "reasoning_skipped",
            "messages": [AIMessage(content="Skip reasoning for out-of-scope query.")],
        }

    query = state["query"]
    planner_result = state["planner_result"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[Step 3] Reasoning agent running... attempt={retry_count + 1}")

    reasoning_agent = ReasoningGraphAgent()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(reasoning_agent.run, query, planner_result)

    try:
        reasoning_result = future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        error_msg = f"Reasoning timeout on attempt {retry_count + 1}"
        print(f"[Degraded] {error_msg}")
        return {
            "reasoning_result": _build_degraded_reasoning_result(error_msg),
            "reasoning_degraded": True,
            "step": "reasoning_completed",
            "messages": [AIMessage(content=error_msg)],
        }
    except Exception as e:
        executor.shutdown(wait=False)
        error_msg = f"Reasoning execution error: {str(e)}"
        print(f"[Degraded] {error_msg}")
        return {
            "reasoning_result": _build_degraded_reasoning_result(error_msg),
            "reasoning_degraded": True,
            "step": "reasoning_completed",
            "messages": [AIMessage(content=error_msg)],
        }
    finally:
        executor.shutdown(wait=False)

    if reasoning_result.get("success"):
        print(
            f"[OK] Reasoning finished. rounds={reasoning_result.get('retrieve_rounds')}, "
            f"tools={reasoning_result.get('called_tools')}"
        )
    else:
        print("[Warn] Reasoning returned non-success payload, but workflow will continue.")

    return {
        "reasoning_result": reasoning_result,
        "reasoning_degraded": not reasoning_result.get("success", False),
        "step": "reasoning_completed",
        "messages": [AIMessage(content="Reasoning completed")],
    }


def evaluation_node(state: ControllerState) -> dict:
    """Patched evaluation node with degraded fallback instead of hard failure."""
    query = state["query"]
    reasoning_result = state["reasoning_result"]
    raw_context = state.get("evaluation_context") or {}
    evaluation_context = dict(raw_context) if isinstance(raw_context, dict) else {}
    intent_payload = state.get("intent_result", {}) or {}
    intent_data = intent_payload.get("data", {}) if isinstance(intent_payload, dict) else {}
    if intent_data:
        evaluation_context.setdefault("domain_scope", intent_data.get("domain_scope"))
        evaluation_context.setdefault("answer_scope_hint", intent_data.get("answer_scope_hint"))
        evaluation_context.setdefault("allowed_tools", intent_data.get("allowed_tools", []))
    if state.get("out_of_scope"):
        evaluation_context.setdefault("domain_scope", "out_of_scope")
        evaluation_context.setdefault("boundary_reason", state.get("boundary_reason"))

    print("\n[Step 4] Evaluation agent running...")

    evaluation_agent = EvaluationGraphAgent()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(evaluation_agent.evaluate, query, reasoning_result, evaluation_context)

    try:
        evaluation_result = future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        error_msg = "Evaluation timeout"
        print(f"[Degraded] {error_msg}")
        return {
            "evaluation_result": _build_degraded_evaluation_result(error_msg),
            "optimization_suggestions": [],
            "evaluation_degraded": True,
            "step": "evaluation_completed",
            "messages": [AIMessage(content=error_msg)],
        }
    except Exception as e:
        executor.shutdown(wait=False)
        error_msg = f"Evaluation execution error: {str(e)}"
        print(f"[Degraded] {error_msg}")
        return {
            "evaluation_result": _build_degraded_evaluation_result(error_msg),
            "optimization_suggestions": [],
            "evaluation_degraded": True,
            "step": "evaluation_completed",
            "messages": [AIMessage(content=error_msg)],
        }
    finally:
        executor.shutdown(wait=False)

    if evaluation_result.get("success"):
        data = evaluation_result.get("data", {})
        optimization_engine = OptimizationEngine()
        analysis = optimization_engine.analyze_evaluation(evaluation_result)
        suggestions = [a["description"] for a in analysis.get("optimization_actions", [])]
        print(
            f"[OK] Evaluation finished. relevance={data.get('retrieval_relevance')}, "
            f"accuracy={data.get('answer_accuracy')}, completeness={data.get('answer_completeness')}"
        )
        return {
            "evaluation_result": evaluation_result,
            "optimization_suggestions": suggestions,
            "optimization_analysis": analysis,
            "evaluation_degraded": False,
            "step": "evaluation_completed",
            "messages": [AIMessage(content="Evaluation completed")],
        }

    print("[Warn] Evaluation returned non-success payload, skip optimization retry.")
    return {
        "evaluation_result": evaluation_result,
        "optimization_suggestions": [],
        "evaluation_degraded": True,
        "step": "evaluation_completed",
        "messages": [AIMessage(content="Evaluation degraded")],
    }


def optimization_node(state: ControllerState) -> dict:
    """Patched optimization node that skips retries for degraded or out-of-scope cases."""
    if state.get("out_of_scope"):
        print("\n[Step 5] Out-of-scope query, skip optimization.")
        return {
            "optimization_applied": False,
            "step": "optimization_skipped",
            "messages": [AIMessage(content="Skip optimization for out-of-scope query.")],
        }

    if state.get("evaluation_degraded"):
        print("\n[Step 5] Evaluation degraded, skip optimization retry.")
        return {
            "optimization_applied": False,
            "step": "optimization_skipped",
            "messages": [AIMessage(content="Skip optimization because evaluation degraded.")],
        }

    intent_payload = state.get("intent_result", {}) or {}
    intent_data = intent_payload.get("data", {}) if isinstance(intent_payload, dict) else {}
    if intent_data.get("domain_scope") == "mixed":
        print("\n[Step 5] Mixed-scope query, skip optimization retry.")
        return {
            "optimization_applied": False,
            "step": "optimization_skipped",
            "messages": [AIMessage(content="Skip optimization for mixed-scope query.")],
        }

    evaluation_result = state.get("evaluation_result", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    data = evaluation_result.get("data", {})
    evaluation_context = state.get("evaluation_context") or {}
    if not isinstance(evaluation_context, dict):
        evaluation_context = {}
    reasoning_payload = state.get("reasoning_result", {}) or {}
    called_tools = reasoning_payload.get("called_tools", []) if isinstance(reasoning_payload, dict) else []
    if not isinstance(called_tools, list):
        called_tools = []

    if data.get("evaluation_mode") == "boundary":
        print("\n[Step 5] Boundary-mode evaluation, skip optimization retry.")
        return {
            "optimization_applied": False,
            "step": "optimization_skipped",
            "messages": [AIMessage(content="Skip optimization for boundary-mode evaluation.")],
        }

    scores = [data.get(key, 0) for key in NUMERIC_SCORE_KEYS]
    avg_score = sum(scores) / len(scores) if scores else 0
    answer_accuracy = data.get("answer_accuracy", 0)
    answer_completeness = data.get("answer_completeness", 0)
    retrieval_relevance = data.get("retrieval_relevance", 0)
    tool_call_appropriateness = data.get("tool_call_appropriateness", 0)

    expected_tool = str(evaluation_context.get("expected_tool") or "").strip()
    category = str(evaluation_context.get("category") or "").strip()

    core_answer_strong = answer_accuracy >= 4 and answer_completeness >= 4
    retrieval_strong = retrieval_relevance >= 4
    severe_answer_issue = answer_accuracy < 4 or answer_completeness < 4
    severe_retrieval_issue = retrieval_relevance < 3

    tool_mismatch = False
    if expected_tool:
        expected_tools = {item.strip() for item in expected_tool.split("|") if item.strip()}
        tool_mismatch = not expected_tools.intersection(set(called_tools))

    tool_retry_worthy = bool(expected_tool) and tool_mismatch and tool_call_appropriateness < 3
    high_quality_answer = core_answer_strong and retrieval_strong
    effective_max_retries = min(max_retries, 1)

    if high_quality_answer and not tool_retry_worthy:
        print(
            "\n[Step 5] Skip optimization. "
            f"High-quality answer detected (avg_score={avg_score:.2f}, expected_tool={expected_tool or 'NONE'}, category={category or 'UNKNOWN'})."
        )
        return {
            "optimization_applied": False,
            "step": "optimization_skipped",
            "messages": [AIMessage(content="Skip optimization because answer quality is already high.")],
        }

    should_retry = retry_count < effective_max_retries and (
        severe_answer_issue
        or severe_retrieval_issue
        or tool_retry_worthy
        or avg_score < 3.4
    )

    if should_retry:
        print(
            f"\n[Step 5] Apply optimization retry. avg_score={avg_score:.2f}, retry={retry_count + 1}, "
            f"tool_retry_worthy={tool_retry_worthy}, severe_answer_issue={severe_answer_issue}, "
            f"severe_retrieval_issue={severe_retrieval_issue}"
        )
        optimization_engine = OptimizationEngine()
        analysis = state.get("optimization_analysis") or optimization_engine.analyze_evaluation(evaluation_result)
        original_plan = state.get("planner_result", {})
        improved_plan = optimization_engine.generate_improved_plan(original_plan, analysis)
        return {
            "planner_result": improved_plan,
            "retry_count": retry_count + 1,
            "optimization_applied": True,
            "baseline_evaluation": state.get("baseline_evaluation") or evaluation_result,
            "optimization_analysis": analysis,
            "optimization_actions_applied": analysis.get("recommended_actions", []),
            "step": "optimization_applied",
            "messages": [AIMessage(content="Optimization retry applied.")],
        }

    print(
        f"\n[Step 5] Skip optimization. avg_score={avg_score:.2f}, retry_count={retry_count}, "
        f"tool_retry_worthy={tool_retry_worthy}, severe_answer_issue={severe_answer_issue}, "
        f"severe_retrieval_issue={severe_retrieval_issue}"
    )
    return {
        "optimization_applied": False,
        "step": "optimization_skipped",
        "messages": [AIMessage(content="Optimization skipped.")],
    }


def after_intent(state: ControllerState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("out_of_scope"):
        return "evaluation"
    return "planner"


def create_controller_agent():
    workflow = StateGraph(ControllerState)

    workflow.add_node("intent", intent_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("optimization", optimization_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.add_edge(START, "intent")

    workflow.add_conditional_edges(
        "intent",
        after_intent,
        {
            "planner": "planner",
            "evaluation": "evaluation",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "planner",
        after_planner,
        {
            "reasoning": "reasoning",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "reasoning",
        after_reasoning,
        {
            "evaluation": "evaluation",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "evaluation",
        after_evaluation,
        {
            "optimization": "optimization",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "optimization",
        after_optimization,
        {
            "reasoning": "reasoning",
            "finalize": "finalize",
            "error_handler": "error_handler",
        },
    )

    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
