#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   intent_agent_graph.py
@Time    :   2026/03/15 21:23:11
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   意图理解代理：使用LangGraph工作流实现
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import re
import json
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

# 定义意图结果结构
class IntentResult(BaseModel):
    intent_type: str = Field(description="rag_search/web_search/weather/unknown")
    domain: str = Field(description="食用菌子领域：香菇/金针菇/杏鲍菇/其他食用菌")
    keywords: List[str] = Field(description="核心关键词列表")
    complexity: str = Field(description="问题复杂度：simple/complex/multi_turn")
    sub_tasks: List[str] = Field(description="拆解后的子任务列表")
    city: Optional[str] = Field(default=None, description="关联城市（天气相关）")

# 定义状态
class IntentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str              # 原始用户问题
    rule_based_result: dict      # 规则解析的中间结果
    llm_based_result: dict       # LLM解析的中间结果
    final_intent: dict           # 最终意图解析结果
    error: Optional[str]         # 错误信息
    step: str                    # 当前步骤

# 初始化LLM
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-32B-Instruct"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0.1
)

class IntentGraphAgent:
    def __init__(self):
        # 规则匹配配置
        self.intent_keywords = {
            "rag_search": ["种植", "温度", "湿度", "培养基", "周期", "通风", "配方"],
            "web_search": ["最新", "市场价格", "政策", "2025", "新技术", "行情"],
            "weather": ["天气", "气温", "降雨", "寒潮", "高温", "气候"]
        }
        self.domain_keywords = {
            "香菇": ["香菇", "香蕈", "冬菇"],
            "金针菇": ["金针菇", "毛柄金钱菌"],
            "杏鲍菇": ["杏鲍菇", "刺芹侧耳"]
        }
        self.complexity_keywords = {
            "complex": ["哪些", "多少", "如何控制", "怎么调节", "条件"],
            "multi_turn": ["为什么", "怎么办", "如何解决", "多次", "步骤"]
        }

    def extract_city(self, query: str) -> Optional[str]:
        """提取问题中的城市名"""
        city_pattern = r"北京|上海|广州|深圳|杭州|成都|重庆|南京|武汉|西安|郑州|济南|青岛"
        match = re.search(city_pattern, query)
        return match.group() if match else None

    def _extract_intent_by_rule(self, query: str) -> str:
        """规则提取意图类型"""
        for intent, keys in self.intent_keywords.items():
            if any(key in query for key in keys):
                return intent
        return "unknown"

    def _extract_domain_by_rule(self, query: str) -> str:
        """规则提取子领域"""
        for domain, keys in self.domain_keywords.items():
            if any(key in query for key in keys):
                return domain
        return "其他食用菌"

    def _extract_keywords_by_rule(self, query: str) -> List[str]:
        """规则拆分关键词"""
        city = self.extract_city(query)
        if city:
            query = query.replace(city, "")
        keyword_pattern = r"香菇|金针菇|杏鲍菇|温度|湿度|种植|周期|通风|培养基|天气|气候"
        keywords = re.findall(keyword_pattern, query)
        if len(keywords) < 2:
            fallback = re.split(r"的|需要|控制|哪些|和", query)
            keywords = [k.strip() for k in fallback if k.strip()][:2]
        return keywords[:5]

    def _extract_complexity_by_rule(self, query: str) -> str:
        """规则提取复杂度"""
        if any(key in query for key in self.complexity_keywords["multi_turn"]):
            return "multi_turn"
        elif any(key in query for key in self.complexity_keywords["complex"]):
            return "complex"
        else:
            return "simple"

    def _split_sub_tasks(self, query: str, complexity: str) -> List[str]:
        """规则拆分子任务"""
        if complexity == "simple":
            return []
        tasks = []
        if "温度" in query and "湿度" in query:
            city = self.extract_city(query) or ""
            domain = self._extract_domain_by_rule(query)
            tasks.append(f"{city}{domain}种植适宜温度")
            tasks.append(f"{city}{domain}种植适宜湿度")
        return tasks[:3]

def rule_based_parse_node(state: IntentState) -> dict:
    """节点1：基于规则的意图解析"""
    agent = IntentGraphAgent()
    user_query = state["messages"][-1].content if state.get("messages") else state.get("user_query", "")
    
    # 规则解析
    intent_type = agent._extract_intent_by_rule(user_query)
    domain = agent._extract_domain_by_rule(user_query)
    keywords = agent._extract_keywords_by_rule(user_query)
    complexity = agent._extract_complexity_by_rule(user_query)
    sub_tasks = agent._split_sub_tasks(user_query, complexity)
    city = agent.extract_city(user_query)

    rule_result = {
        "intent_type": intent_type,
        "domain": domain,
        "keywords": keywords,
        "complexity": complexity,
        "sub_tasks": sub_tasks,
        "city": city
    }

    return {
        "user_query": user_query,
        "rule_based_result": rule_result,
        "step": "rule_parsed",
        "messages": [AIMessage(content=f"规则解析完成：意图类型={intent_type}")]
    }

def llm_based_parse_node(state: IntentState) -> dict:
    """节点2：LLM-based意图解析（增强解析）"""
    user_query = state["user_query"]
    
    prompt = f"""你是一个专业的意图解析助手。请分析用户的问题并返回JSON格式的解析结果。

用户问题：{user_query}

必须包含以下字段：
- intent_type: rag_search/web_search/weather/unknown
- domain: 香菇/金针菇/杏鲍菇/其他食用菌
- keywords: 核心关键词列表（最多5个）
- complexity: simple/complex/multi_turn
- sub_tasks: 拆解后的子任务列表（如果没有子任务则返回空列表）
- city: 关联城市（如果没有则返回null）

请确保返回合法的JSON格式。"""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        response_text = response.content
        
        # 提取JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # 清理可能的注释和格式问题
            json_str = re.sub(r'//.*?\n', '', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            
            llm_result = json.loads(json_str)
            
            # 确保所有字段都存在
            required_fields = ["intent_type", "domain", "keywords", "complexity", "sub_tasks", "city"]
            for field in required_fields:
                if field not in llm_result:
                    llm_result[field] = None if field == "city" else ([] if field in ["keywords", "sub_tasks"] else "unknown")
            
            return {
                "llm_based_result": llm_result,
                "step": "llm_parsed",
                "messages": [AIMessage(content="LLM解析完成")]
            }
        else:
            raise ValueError("No JSON found in response")
            
    except Exception as e:
        return {
            "llm_based_result": None,
            "error": f"LLM解析失败：{str(e)}",
            "step": "llm_parse_failed",
            "messages": [AIMessage(content=f"LLM解析失败，将使用规则解析结果")]
        }

def merge_results_node(state: IntentState) -> dict:
    """节点3：合并规则和LLM的解析结果"""
    rule_result = state.get("rule_based_result", {})
    llm_result = state.get("llm_based_result")
    error = state.get("error")
    
    # 决策逻辑：优先使用LLM结果（如果可用且合理），否则使用规则结果
    if llm_result and not error:
        # 验证LLM结果的合理性
        if llm_result.get("intent_type") in ["rag_search", "web_search", "weather", "unknown"]:
            final_result = llm_result
            merge_message = "使用LLM解析结果"
        else:
            final_result = rule_result
            merge_message = "LLM结果不合理，回退到规则解析"
    else:
        final_result = rule_result
        merge_message = "使用规则解析结果"
    
    # 确保城市字段存在
    if "city" not in final_result or final_result["city"] is None:
        # 尝试从规则结果中获取
        if rule_result.get("city"):
            final_result["city"] = rule_result["city"]
        else:
            # 最后尝试从原始查询中提取
            agent = IntentGraphAgent()
            final_result["city"] = agent.extract_city(state["user_query"])
    
    return {
        "final_intent": final_result,
        "step": "merged",
        "messages": [AIMessage(content=f"解析完成：{merge_message}")]
    }

def validate_result_node(state: IntentState) -> dict:
    """节点4：验证最终结果"""
    final_intent = state.get("final_intent", {})
    
    try:
        # 尝试创建IntentResult对象进行验证
        validated_result = IntentResult(**final_intent)
        
        return {
            "final_intent": validated_result.model_dump(),
            "step": "validated",
            "messages": [AIMessage(content="✅ 意图解析验证通过")]
        }
    except Exception as e:
        # 如果验证失败，创建默认的兜底结果
        agent = IntentGraphAgent()
        user_query = state["user_query"]
        
        fallback_result = IntentResult(
            intent_type="rag_search",
            domain=agent._extract_domain_by_rule(user_query),
            keywords=agent._extract_keywords_by_rule(user_query),
            complexity="simple",
            sub_tasks=[],
            city=agent.extract_city(user_query)
        )
        
        return {
            "final_intent": fallback_result.model_dump(),
            "error": f"验证失败，使用兜底结果：{str(e)}",
            "step": "validated_with_fallback",
            "messages": [AIMessage(content="⚠️ 使用兜底解析结果")]
        }

def error_handler_node(state: IntentState) -> dict:
    """错误处理节点"""
    error = state.get("error", "未知错误")
    user_query = state.get("user_query", "")
    agent = IntentGraphAgent()
    
    fallback_result = IntentResult(
        intent_type="rag_search",
        domain=agent._extract_domain_by_rule(user_query),
        keywords=agent._extract_keywords_by_rule(user_query) or ["查询"],
        complexity="simple",
        sub_tasks=[],
        city=agent.extract_city(user_query)
    )
    
    return {
        "final_intent": fallback_result.model_dump(),
        "error": f"处理失败：{error}",
        "step": "error_handled",
        "messages": [AIMessage(content="❌ 处理出错，返回兜底结果")]
    }

def should_use_llm(state: IntentState) -> str:
    """条件边：决定是否使用LLM解析"""
    rule_result = state.get("rule_based_result", {})
    
    # 如果规则解析结果置信度低（比如unknown意图），则使用LLM增强
    if rule_result.get("intent_type") == "unknown" or not rule_result.get("keywords"):
        return "llm_based_parse"
    return "merge_results"

def should_validate(state: IntentState) -> str:
    """条件边：决定是否进入验证节点"""
    if state.get("error"):
        return "error_handler"
    return "validate_result"

def create_intent_agent():
    """创建工作流图"""
    
    # 创建工作流
    workflow = StateGraph(IntentState)
    
    # 添加节点
    workflow.add_node("rule_based_parse", rule_based_parse_node)
    workflow.add_node("llm_based_parse", llm_based_parse_node)
    workflow.add_node("merge_results", merge_results_node)
    workflow.add_node("validate_result", validate_result_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # 设置入口
    workflow.add_edge(START, "rule_based_parse")
    
    # 规则解析后，根据置信度决定是否使用LLM
    workflow.add_conditional_edges(
        "rule_based_parse",
        should_use_llm,  # 只在这里使用条件边
        {
            "llm_based_parse": "llm_based_parse",
            "merge_results": "merge_results"
        }
    )
    
    # 如果走了LLM分支，解析后直接合并
    workflow.add_edge("llm_based_parse", "merge_results")
    
    # 合并后验证
    workflow.add_edge("merge_results", "validate_result")
    
    # 验证后：成功则结束，失败则错误处理
    workflow.add_conditional_edges(
        "validate_result",
        lambda state: "error_handler" if state.get("error") else END,
        {
            "error_handler": "error_handler",
            END: END
        }
    )
    
    # 错误处理后结束
    workflow.add_edge("error_handler", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def parse_intent(query: str) -> dict:
    """解析用户意图的主函数"""
    app = create_intent_agent()
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "step": "start",
        "error": None
    }
    
    try:
        result = app.invoke(
            inputs,
            config={"configurable": {"thread_id": f"intent-{hash(query)}"}}
        )
        
        return {
            "success": True,
            "data": result.get("final_intent", {}),
            "step": result.get("step", "completed"),
            "error": result.get("error")
        }
    except Exception as e:
        # 如果图执行失败，使用简单的规则解析作为兜底
        agent = IntentGraphAgent()
        fallback = {
            "intent_type": agent._extract_intent_by_rule(query),
            "domain": agent._extract_domain_by_rule(query),
            "keywords": agent._extract_keywords_by_rule(query),
            "complexity": agent._extract_complexity_by_rule(query),
            "sub_tasks": agent._split_sub_tasks(query, agent._extract_complexity_by_rule(query)),
            "city": agent.extract_city(query)
        }
        
        return {
            "success": False,
            "data": fallback,
            "error": f"工作流执行失败：{str(e)}，使用规则兜底"
        }

if __name__ == "__main__":
    # 测试代码
    test_queries = "北京种植香菇需要控制哪些温度和湿度条件？"
    result = parse_intent(test_queries)
    print(json.dumps(result, ensure_ascii=False, indent=2))