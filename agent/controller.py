#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   controller_with_optimization.py
@Time    :   2026/03/15 23:45:00
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   增强版控制器：根据评估结果自动优化系统
"""

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
    optimization_suggestions: List[str]      # 优化建议列表
    optimization_applied: bool                # 是否已应用优化
    retry_count: int                          # 重试次数
    error: Optional[str]                      # 错误信息
    step: str                                 # 当前步骤
    thread_id: Optional[str]                  # 线程ID


# ========== 优化规则引擎 ==========
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
        
        return {
            "scores": scores,
            "low_score_dimensions": low_score_dimensions,
            "original_suggestion": suggestion,
            "optimization_actions": optimization_actions
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
    
    def generate_improved_plan(self, original_plan: Dict, evaluation_analysis: Dict) -> Dict:
        """根据评估分析生成改进的检索计划"""
        
        low_score_dims = evaluation_analysis.get("low_score_dimensions", [])
        actions = evaluation_analysis.get("optimization_actions", [])
        
        # 复制原始计划
        improved_plan = original_plan.copy() if original_plan else {"data": {}}
        
        # 确保有data字段
        if "data" not in improved_plan:
            improved_plan["data"] = {}
        
        plan_data = improved_plan["data"]
        
        # 根据低分维度自动调整参数
        if "retrieval_relevance" in low_score_dims:
            # 检索相关性低：增加top_k，启用多轮检索
            plan_data["top_k"] = min(plan_data.get("top_k", 5) + 2, 10)
            plan_data["multi_round"] = True
            print(f"📊 优化：检索相关性低，top_k调整为{plan_data['top_k']}，启用多轮检索")
        
        if "answer_completeness" in low_score_dims:
            # 答案完整性低：增加检索轮次，扩展关键词
            plan_data["multi_round"] = True
            if "expand_keywords" not in plan_data:
                plan_data["expand_keywords"] = []
            plan_data["expand_keywords"].extend(["详细", "完整", "全部"])
            print(f"📊 优化：答案完整性低，启用多轮检索，添加扩展关键词")
        
        if "tool_call_appropriateness" in low_score_dims:
            # 工具调用问题：强制在下次尝试中调用工具
            plan_data["force_tool_call"] = True
            print(f"📊 优化：工具调用问题，下次将强制评估工具需求")
        
        if "result_fusion_quality" in low_score_dims:
            # 结果融合质量低：增加rerank
            plan_data["rerank"] = True
            print(f"📊 优化：结果融合质量低，启用重排序")
        
        if "answer_optimization_effect" in low_score_dims:
            # 答案优化效果低：在优化阶段提供更详细的上下文
            plan_data["detailed_optimization"] = True
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
        
        return improved_plan
    
    def generate_optimization_report(self, before: Dict, after: Dict, evaluation: Dict) -> str:
        """生成优化前后对比报告"""
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
    future = executor.submit(evaluation_agent.evaluate, query, reasoning_result)
    
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
    max_retries = 2  # 最多重试2次
    
    data = evaluation_result.get("data", {})
    
    # 计算平均分数
    scores = [
        data.get("retrieval_relevance", 0),
        data.get("answer_accuracy", 0),
        data.get("answer_completeness", 0),
        data.get("reasoning_effectiveness", 0)
    ]
    avg_score = sum(scores) / len(scores)
    
    # 决策逻辑：如果平均分低于3.5且还有重试次数，则优化重试
    if avg_score < 3.5 and retry_count < max_retries:
        print(f"\n🔄 [步骤5] 评估分数较低 ({avg_score:.1f}/5)，启动优化重试 (第{retry_count + 1}次)")
        
        # 根据评估结果调整规划
        optimization_engine = OptimizationEngine()
        analysis = optimization_engine.analyze_evaluation(evaluation_result)
        
        # 生成改进的规划
        original_plan = state.get("planner_result", {})
        improved_plan = optimization_engine.generate_improved_plan(original_plan, analysis)
        
        return {
            "planner_result": improved_plan,
            "retry_count": retry_count + 1,
            "optimization_applied": True,
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
        # 这里可以生成前后对比报告
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
    
    def run(self, query: str, max_retries: int = 2) -> Dict:
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
            "retry_count": 0,
            "optimization_suggestions": [],
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
            final_score = sum([
                eval_data.get("retrieval_relevance", 0),
                eval_data.get("answer_accuracy", 0),
                eval_data.get("answer_completeness", 0)
            ]) / 3 if eval_data else 0

            return {
                "success": not result.get("error"),
                "query": query,
                "retry_count": result.get("retry_count", 0),
                "final_score": round(final_score, 1),
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
                "step": result.get("step", "completed"),
                "error": result.get("error")
            }
            
        except Exception as e:
            error_msg = f"控制器工作流执行失败：{str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "query": query,
                "error": error_msg
            }


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
    test_queries = "北京种植香菇需要控制哪些温度和湿度条件？"
    result = controller.run(test_queries, max_retries=2)
    print_optimized_result(result)