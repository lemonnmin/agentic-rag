#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   retrieval_planner_agent_graph.py
@Time    :   2026/03/15 21:23:11
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   检索规划代理：使用LangGraph工作流实现（修复版）
"""

from typing import TypedDict, Annotated, Dict, List, Optional
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
import json
import re
from dotenv import load_dotenv

# 加载环境变量
def load_env_from_root():
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

# 检索策略结构
class RetrievalPlan(BaseModel):
    retrievers: List[str] = Field(default=["vector"], description="检索器列表：vector/bm25/hybrid")
    top_k: int = Field(default=3, description="返回结果数")
    rerank: bool = Field(default=True, description="是否重排序")
    multi_round: bool = Field(default=False, description="是否多轮检索")
    expand_keywords: List[str] = Field(default=[], description="扩展关键词")
    mode: str = Field(default="normal", description="检索模式")
    sub_tasks: List[str] = Field(default=[], description="检索子任务")

# 初始化LLM（用于智能策略生成）
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-32B-Instruct"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0.1
)

# 定义状态
class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    intent_data: Dict                # 输入的意图数据
    rule_based_plan: Optional[Dict]  # 规则生成的策略
    llm_based_plan: Optional[Dict]   # LLM生成的策略
    final_plan: Optional[Dict]       # 最终策略
    error: Optional[str]              # 错误信息
    step: str                         # 当前步骤

class RetrievalPlannerGraphAgent:
    def __init__(self):
        # 策略映射表
        self.strategy_map = {
            # 简单RAG问题：向量检索+小top_k + normal模式
            ("rag_search", "simple"): RetrievalPlan(
                retrievers=["vector"],
                top_k=3,
                rerank=True,
                multi_round=False,
                mode="simple"
            ),
            # 复杂RAG问题：混合检索+大top_k+关键词扩展 + accurate模式
            ("rag_search", "complex"): RetrievalPlan(
                retrievers=["vector", "bm25"],
                top_k=5,
                rerank=True,
                multi_round=True,
                expand_keywords=["种植技术", "环境控制"],
                mode="complex"
            ),
            # 多轮问题：混合检索+多轮 + accurate模式
            ("rag_search", "multi_turn"): RetrievalPlan(
                retrievers=["hybrid"],
                top_k=4,
                rerank=True,
                multi_round=True,
                mode="multi_turn"
            ),
            # 天气问题：仅扩展关键词 + fast模式
            ("weather", "*"): RetrievalPlan(
                retrievers=["vector"],
                top_k=2,
                rerank=False,
                multi_round=False,
                expand_keywords=["气候适应", "温度调节"],
                mode="simple"
            )
        }

    def _safe_get_intent_data(self, intent_data: Dict) -> Dict:
        """安全解析意图数据"""
        if isinstance(intent_data, dict):
            if intent_data.get("success") is False:
                return intent_data.get("data", {})
            return intent_data.get("data", {})
        return {}

def parse_intent_node(state: PlannerState) -> dict:
    """节点1：解析输入的意图数据"""
    intent_data = state.get("intent_data", {})
    
    # 安全解析
    agent = RetrievalPlannerGraphAgent()
    intent_core = agent._safe_get_intent_data(intent_data)
    
    return {
        "intent_data": intent_core,
        "step": "intent_parsed",
        "messages": [AIMessage(content=f"意图解析完成：类型={intent_core.get('intent_type')}, 复杂度={intent_core.get('complexity')}")]
    }

def rule_based_planning_node(state: PlannerState) -> dict:
    """节点2：基于规则的策略生成"""
    intent_core = state.get("intent_data", {})
    agent = RetrievalPlannerGraphAgent()
    
    intent_type = intent_core.get("intent_type", "rag_search")
    complexity = intent_core.get("complexity", "simple")
    domain = intent_core.get("domain", "通用")
    
    # 策略匹配
    key_exact = (intent_type, complexity)
    key_fallback = (intent_type, "*")
    
    if key_exact in agent.strategy_map:
        plan = agent.strategy_map[key_exact]
    elif key_fallback in agent.strategy_map:
        plan = agent.strategy_map[key_fallback]
    else:
        # 终极兜底策略
        plan = RetrievalPlan(
            retrievers=["vector"],
            top_k=3,
            mode="normal"
        )
    
    # 转换为字典
    rule_plan = plan.dict()
    
    # 添加领域关键词
    expand_keywords = []
    if domain and domain != "通用":
        expand_keywords.append(domain)
    
    # 添加意图关键词
    intent_keywords = intent_core.get("keywords", [])
    if isinstance(intent_keywords, list):
        expand_keywords.extend(intent_keywords)
    
    # 添加策略默认关键词
    expand_keywords.extend(rule_plan.get("expand_keywords", []))
    
    # 去重
    rule_plan["expand_keywords"] = list(dict.fromkeys(expand_keywords))
    rule_plan["sub_tasks"] = intent_core.get("sub_tasks", [])
    
    return {
        "rule_based_plan": rule_plan,
        "step": "rule_planned",
        "messages": [AIMessage(content=f"规则策略生成完成：模式={rule_plan.get('mode')}")]
    }

def llm_based_planning_node(state: PlannerState) -> dict:
    """节点3：LLM-based策略生成（智能优化）"""
    intent_core = state.get("intent_data", {})
    
    prompt = f"""你是一个检索策略规划专家。请根据用户意图数据，生成最优的检索策略。

意图数据：
- 意图类型：{intent_core.get('intent_type')}
- 领域：{intent_core.get('domain')}
- 关键词：{intent_core.get('keywords')}
- 复杂度：{intent_core.get('complexity')}
- 子任务：{intent_core.get('sub_tasks')}
- 城市：{intent_core.get('city')}

请返回JSON格式的检索策略，包含以下字段：
- retrievers: 检索器列表 ["vector", "bm25", "hybrid"] 中的一个或多个
- top_k: 返回结果数（整数，3-10之间）
- rerank: 是否重排序（boolean）
- multi_round: 是否多轮检索（boolean）
- expand_keywords: 扩展关键词列表（数组）
- mode: 检索模式（simple/complex/multi_turn）
- sub_tasks: 子任务列表（从意图中继承）

请根据问题复杂度选择合适的参数：
- simple问题：top_k较小(3-4)，不需要多轮检索
- complex问题：top_k较大(5-7)，可能需要多轮检索和重排序
- multi_turn问题：需要多轮检索能力

请确保返回合法的JSON格式。"""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        response_text = response.content
        
        # 提取JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # 清理可能的格式问题
            json_str = re.sub(r',\s*}', '}', json_str)
            
            llm_plan = json.loads(json_str)
            
            # 确保必要字段存在
            required_fields = ["retrievers", "top_k", "rerank", "multi_round", "expand_keywords", "mode", "sub_tasks"]
            for field in required_fields:
                if field not in llm_plan:
                    if field == "retrievers":
                        llm_plan[field] = ["vector"]
                    elif field in ["top_k"]:
                        llm_plan[field] = 3
                    elif field in ["rerank", "multi_round"]:
                        llm_plan[field] = False
                    elif field in ["expand_keywords", "sub_tasks"]:
                        llm_plan[field] = []
                    else:
                        llm_plan[field] = "normal"
            
            return {
                "llm_based_plan": llm_plan,
                "step": "llm_planned",
                "messages": [AIMessage(content="LLM策略生成完成")]
            }
        else:
            raise ValueError("No JSON found in response")
            
    except Exception as e:
        return {
            "llm_based_plan": None,
            "error": f"LLM策略生成失败：{str(e)}",
            "step": "llm_plan_failed",
            "messages": [AIMessage(content=f"LLM策略生成失败，将使用规则策略")]
        }

def merge_plans_node(state: PlannerState) -> dict:
    """节点4：合并规则和LLM的策略"""
    rule_plan = state.get("rule_based_plan", {})
    llm_plan = state.get("llm_based_plan")
    error = state.get("error")
    
    # 决策逻辑：如果LLM成功且有合理结果，使用LLM策略；否则使用规则策略
    if llm_plan and not error:
        # 验证LLM结果的合理性
        if isinstance(llm_plan.get("top_k"), int) and 1 <= llm_plan.get("top_k", 0) <= 10:
            final_plan = llm_plan
            merge_message = "使用LLM生成的优化策略"
        else:
            final_plan = rule_plan
            merge_message = "LLM结果不合理，回退到规则策略"
    else:
        final_plan = rule_plan
        merge_message = "使用规则策略"
    
    # 确保子任务存在
    if "sub_tasks" not in final_plan or not final_plan["sub_tasks"]:
        final_plan["sub_tasks"] = state.get("intent_data", {}).get("sub_tasks", [])
    
    return {
        "final_plan": final_plan,
        "step": "plans_merged",
        "messages": [AIMessage(content=f"策略合并完成：{merge_message}")]
    }

def validate_plan_node(state: PlannerState) -> dict:
    """节点5：验证最终策略"""
    final_plan = state.get("final_plan", {})
    
    try:
        # 尝试创建RetrievalPlan对象进行验证
        validated_plan = RetrievalPlan(**final_plan)
        
        return {
            "final_plan": validated_plan.dict(),
            "step": "plan_validated",
            "messages": [AIMessage(content="✅ 检索策略验证通过")]
        }
    except Exception as e:
        # 如果验证失败，创建默认的兜底策略
        fallback_plan = RetrievalPlan()
        
        return {
            "final_plan": fallback_plan.dict(),
            "error": f"验证失败，使用兜底策略：{str(e)}",
            "step": "plan_validated_with_fallback",
            "messages": [AIMessage(content="⚠️ 使用兜底检索策略")]
        }

def final_node(state: PlannerState) -> dict:
    """最终节点：准备返回结果"""
    return {
        "step": "completed",
        "messages": [AIMessage(content="检索策略生成完成")]
    }

def error_handler_node(state: PlannerState) -> dict:
    """错误处理节点"""
    error = state.get("error", "未知错误")
    
    fallback_plan = RetrievalPlan()
    
    return {
        "final_plan": fallback_plan.dict(),
        "error": f"处理失败：{error}",
        "step": "error_handled",
        "messages": [AIMessage(content="❌ 处理出错，返回兜底策略")]
    }

def should_use_llm(state: PlannerState) -> str:
    """条件边：决定是否使用LLM优化"""
    intent_data = state.get("intent_data", {})
    complexity = intent_data.get("complexity", "simple")
    
    # 复杂问题或需要智能优化的场景使用LLM
    if complexity in ["complex", "multi_turn"]:
        return "llm_based_planning"
    return "merge_plans"

def should_validate(state: PlannerState) -> str:
    """条件边：决定验证后的去向"""
    if state.get("error"):
        return "error_handler"
    return "final"

def create_retrieval_planner_agent():
    """创建工作流图"""
    workflow = StateGraph(PlannerState)
    
    # 添加节点
    workflow.add_node("parse_intent", parse_intent_node)
    workflow.add_node("rule_based_planning", rule_based_planning_node)
    workflow.add_node("llm_based_planning", llm_based_planning_node)
    workflow.add_node("merge_plans", merge_plans_node)
    workflow.add_node("validate_plan", validate_plan_node)
    workflow.add_node("final", final_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # 从开始到意图解析
    workflow.add_edge(START, "parse_intent")
    
    # 意图解析后到规则规划
    workflow.add_edge("parse_intent", "rule_based_planning")
    
    # 规则规划后决定是否用LLM
    workflow.add_conditional_edges(
        "rule_based_planning",
        should_use_llm,
        {
            "llm_based_planning": "llm_based_planning",
            "merge_plans": "merge_plans"
        }
    )
    
    # LLM规划后合并
    workflow.add_edge("llm_based_planning", "merge_plans")
    
    # 合并后验证
    workflow.add_edge("merge_plans", "validate_plan")
    
    # 验证后根据是否有错误决定去向
    workflow.add_conditional_edges(
        "validate_plan",
        should_validate,
        {
            "error_handler": "error_handler",
            "final": "final"
        }
    )
    
    # 错误处理后结束
    workflow.add_edge("error_handler", "final")
    
    # 最终节点结束
    workflow.add_edge("final", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def plan_retrieval(intent_data: Dict) -> Dict:
    """生成检索策略的主函数"""
    try:
        app = create_retrieval_planner_agent()
        
        inputs = {
            "messages": [HumanMessage(content="开始生成检索策略")],
            "intent_data": intent_data,
            "step": "start",
            "error": None
        }
        
        result = app.invoke(
            inputs,
            config={"configurable": {"thread_id": f"planner-{hash(str(intent_data))}"}}
        )
        
        # 确保返回正确的格式
        final_plan = result.get("final_plan", {})
        if not final_plan:
            # 如果final_plan为空，使用默认策略
            agent = RetrievalPlannerGraphAgent()
            intent_core = agent._safe_get_intent_data(intent_data)
            default_plan = RetrievalPlan(
                retrievers=["vector"],
                top_k=3,
                mode="normal",
                sub_tasks=intent_core.get("sub_tasks", [])
            )
            final_plan = default_plan.dict()
        
        return {
            "success": True,
            "data": final_plan,
            "step": result.get("step", "completed"),
            "error": result.get("error")
        }
        
    except Exception as e:
        # 如果图执行失败，使用简单的规则策略作为兜底
        print(f"工作流执行异常: {e}")
        agent = RetrievalPlannerGraphAgent()
        intent_core = agent._safe_get_intent_data(intent_data)
        
        fallback_plan = RetrievalPlan(
            retrievers=["vector"],
            top_k=3,
            mode="normal",
            sub_tasks=intent_core.get("sub_tasks", [])
        )
        
        return {
            "success": False,
            "data": fallback_plan.dict(),
            "error": f"工作流执行失败：{str(e)}，使用规则兜底"
        }

if __name__ == "__main__":
    # 测试代码
    test_intent = {
        "success": True,
        "data": {
            "intent_type": "rag_search",
            "domain": "香菇",
            "keywords": ["种植", "香菇", "温度", "湿度"],
            "complexity": "complex",
            "sub_tasks": ["北京香菇种植适宜温度", "北京香菇种植适宜湿度"],
            "city": "北京"
        },
        "error": None
    }
    
    print("="*50)
    print("测试：复杂RAG问题")
    result = plan_retrieval(test_intent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    