#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   rag_engine_graph.py
@Time    :   2026/03/15 21:23:11
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   RAG引擎：使用LangGraph工作流实现交织推理-检索循环（修复参数问题）
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Optional, TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from dotenv import load_dotenv
from rag.LLM import OpenAIChat
from tools.rag_search import RagSearchTool

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

# 检索结果结构
class RetrievalResult(BaseModel):
    content: str
    source: str
    score: float

# 定义状态 - 移除rag_tool，改为存储必要参数
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str                          # 用户原始问题
    planner_data: Dict                  # 检索规划数据
    rag_mode: str                        # 当前RAG模式
    context: List[str]                   # 上下文列表
    retrieval_results: List[Dict]         # 检索结果列表
    current_round: int                    # 当前检索轮次
    max_rounds: int                       # 最大检索轮次
    need_retrieve: bool                    # 是否需要继续检索
    expand_keywords: List[str]             # 扩展关键词
    top_k: int                             # 返回结果数
    multi_round: bool                      # 是否多轮检索
    final_answer: Optional[str]            # 最终答案
    error: Optional[str]                   # 错误信息
    step: str                              # 当前步骤
    thread_id: Optional[str]               # 线程ID

class RAGEngineGraph:
    def __init__(self):
        load_env_from_root()
        self.llm = OpenAIChat()
        self.rag_tool_cache = {}  # 缓存rag_tool实例，key为thread_id
        
        # 交织推理-检索提示词
        self.reasoning_prompt = ChatPromptTemplate.from_template("""
        你是食用菌种植专家，需完成交织推理-检索循环：
        1. 基于当前上下文和问题，判断是否需要补充检索
        2. 如需补充，返回需要检索的关键词；如无需补充，直接回答问题
        
        上下文：{context}
        问题：{question}
        子任务：{sub_tasks}
        
        【强制格式要求】
        - 必须返回标准JSON字符串，仅包含JSON，无其他文字、注释、markdown格式
        - need_retrieve字段必须是布尔值（True/False）
        - 需补充检索：返回 {{"need_retrieve": True, "keywords": ["关键词1", "关键词2"]}}
        - 无需补充：返回 {{"need_retrieve": False, "answer": "最终答案"}}
        """)
    
    def get_rag_tool(self, thread_id: str, rag_mode: str) -> RagSearchTool:
        """获取或创建RAG工具实例"""
        if thread_id not in self.rag_tool_cache:
            print(f"🔄 创建新的RAG工具实例，模式：{rag_mode}，thread_id：{thread_id}")
            rag_tool = RagSearchTool(mode=rag_mode)
            if rag_mode == "multi_turn":
                rag_tool.clear_history()
            self.rag_tool_cache[thread_id] = rag_tool
        return self.rag_tool_cache[thread_id]

# 创建全局引擎实例
_engine_instance = None

def get_engine():
    """获取全局RAGEngineGraph实例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngineGraph()
    return _engine_instance

def initialize_node(state: RAGState) -> dict:
    """节点1：初始化RAG引擎状态"""
    query = state.get("query", "")
    planner_data = state.get("planner_data", {})
    thread_id = state.get("thread_id", "default")
    
    # 解析planner数据
    if isinstance(planner_data, dict):
        planner_core = planner_data.get("data", planner_data)
    else:
        planner_core = {}
    
    # 确定RAG模式
    rag_mode = planner_core.get("mode")
    if rag_mode is None:
        complexity = planner_core.get("complexity", "simple")
        if complexity == "simple":
            rag_mode = "simple"
        elif complexity == "complex":
            rag_mode = "complex"
        else:
            rag_mode = "multi_turn"
    
    print(f"📌 检测到的rag_mode: {rag_mode}")
    
    # 确定最大检索轮次
    multi_round = planner_core.get("multi_round", False)
    max_rounds = planner_core.get("max_rounds", 2 if multi_round else 1)
    top_k = planner_core.get("top_k", 3)
    
    return {
        "query": query,
        "planner_data": planner_core,
        "rag_mode": rag_mode,
        "context": [],
        "retrieval_results": [],
        "current_round": 0,
        "max_rounds": max_rounds,
        "need_retrieve": True,
        "expand_keywords": planner_core.get("expand_keywords", []),
        "top_k": top_k,
        "multi_round": multi_round,
        "thread_id": thread_id,
        "step": "initialized",
        "messages": [AIMessage(content=f"RAG引擎初始化完成，模式={rag_mode}，最大轮次={max_rounds}")]
    }

def retrieve_node(state: RAGState) -> dict:
    """节点2：执行检索（不使用config参数）"""
    query = state.get("query", "")
    planner_core = state.get("planner_data", {})
    expand_keywords = state.get("expand_keywords", [])
    current_round = state.get("current_round", 0)
    rag_mode = state.get("rag_mode", "unknown")
    top_k = state.get("top_k", 3)
    thread_id = state.get("thread_id", "default")
    
    try:
        # 获取或创建RAG工具
        engine = get_engine()
        rag_tool = engine.get_rag_tool(thread_id, rag_mode)
        
        # 构建当前查询
        current_query = query
        if expand_keywords and current_round > 0:  # 第一轮后加入扩展关键词
            current_query += " " + " ".join(expand_keywords)
        
        print(f"🔍 第{current_round + 1}轮检索，查询词：{current_query}")
        
        # 临时覆盖top_k
        original_top_k = rag_tool.strategy.top_k
        rag_tool.strategy.top_k = top_k
        
        # 执行检索
        tool_docs = rag_tool.retrieve(current_query)
        
        # 恢复原top_k
        rag_tool.strategy.top_k = original_top_k
        
        # 转换为RetrievalResult
        retrieval_results = []
        for idx, doc in enumerate(tool_docs):
            retrieval_results.append({
                "content": doc.strip(),
                "source": f"rag_{rag_mode}",
                "score": 1.0 - (idx / len(tool_docs)) if tool_docs else 0
            })
        
        # 更新上下文
        new_context = state.get("context", [])
        new_context.extend([r["content"] for r in retrieval_results])
        
        # 去重
        unique_content = set()
        unique_results = []
        for res in retrieval_results:
            if res["content"] and res["content"] not in unique_content:
                unique_content.add(res["content"])
                unique_results.append(res)
        
        print(f"✅ 第{current_round + 1}轮检索完成，获取到{len(unique_results)}条结果")
        
        return {
            "context": new_context,
            "retrieval_results": unique_results,
            "current_round": current_round + 1,
            "rag_mode": rag_mode,
            "thread_id": thread_id,
            "step": f"retrieved_round_{current_round + 1}",
            "messages": [AIMessage(content=f"✅ 第{current_round + 1}轮检索完成，获取到{len(unique_results)}条结果")]
        }
    except Exception as e:
        error_msg = f"检索失败：{str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "rag_mode": rag_mode,
            "thread_id": thread_id,
            "step": "retrieve_failed",
            "messages": [AIMessage(content=f"❌ 第{current_round + 1}轮检索失败：{str(e)}")]
        }

def reasoning_node(state: RAGState) -> dict:
    """节点3：推理判断是否需要继续检索"""
    query = state.get("query", "")
    context = state.get("context", [])
    planner_core = state.get("planner_data", {})
    current_round = state.get("current_round", 0)
    rag_mode = state.get("rag_mode", "unknown")
    thread_id = state.get("thread_id", "default")
    
    sub_tasks = planner_core.get("sub_tasks", [])
    context_text = "\n".join(context) if context else "暂无检索结果"
    
    print(f"🧠 第{current_round}轮推理中...")
    
    try:
        # 生成提示词
        prompt_text = f"""
        你是食用菌种植专家，需完成交织推理-检索循环：
        1. 基于当前上下文和问题，判断是否需要补充检索
        2. 如需补充，返回需要检索的关键词；如无需补充，直接回答问题
        
        上下文：{context_text}
        问题：{query}
        子任务：{sub_tasks}
        
        【强制格式要求】
        - 必须返回标准JSON字符串，仅包含JSON，无其他文字、注释、markdown格式
        - need_retrieve字段必须是布尔值（True/False）
        - 需补充检索：返回 {{"need_retrieve": True, "keywords": ["关键词1", "关键词2"]}}
        - 无需补充：返回 {{"need_retrieve": False, "answer": "最终答案"}}
        """
        
        # 调用LLM
        engine = get_engine()
        llm_response = engine.llm.chat(
            prompt=prompt_text,
            history=[],
            content=context_text
        )
        
        print(f"📝 LLM原始返回：{llm_response[:100]}...")
        
        # 解析JSON
        try:
            # 清理响应
            clean_response = llm_response.strip()
            if clean_response.startswith("```") and clean_response.endswith("```"):
                clean_response = clean_response[clean_response.find("{"):clean_response.rfind("}")+1]
            clean_response = clean_response.replace("'", "\"")
            clean_response = clean_response.replace("True", "true")
            clean_response = clean_response.replace("False", "false")
            
            result = json.loads(clean_response)
            
            need_retrieve = result.get("need_retrieve", False)
            keywords = result.get("keywords", [])
            answer = result.get("answer", "")
            
            print(f"✅ 推理结果：need_retrieve={need_retrieve}, keywords={keywords}")
            
            return {
                "need_retrieve": need_retrieve,
                "expand_keywords": keywords,
                "final_answer": answer if not need_retrieve else None,
                "rag_mode": rag_mode,
                "thread_id": thread_id,
                "step": f"reasoned_round_{current_round}",
                "messages": [AIMessage(content=f"推理完成：need_retrieve={need_retrieve}")]
            }
        except (json.JSONDecodeError, ValueError) as e:
            # 解析失败，默认停止检索
            print(f"⚠️ JSON解析失败：{e}")
            return {
                "need_retrieve": False,
                "final_answer": llm_response.strip(),
                "error": f"JSON解析失败：{str(e)}",
                "rag_mode": rag_mode,
                "thread_id": thread_id,
                "step": "reasoning_failed",
                "messages": [AIMessage(content=f"⚠️ 推理结果解析失败，使用原始返回作为答案")]
            }
    except Exception as e:
        print(f"❌ 推理失败：{e}")
        return {
            "need_retrieve": False,
            "error": f"推理失败：{str(e)}",
            "rag_mode": rag_mode,
            "thread_id": thread_id,
            "step": "reasoning_failed",
            "messages": [AIMessage(content=f"❌ 推理失败：{str(e)}")]
        }

def generate_final_answer_node(state: RAGState) -> dict:
    """节点4：生成最终答案（兜底）"""
    query = state.get("query", "")
    context = state.get("context", [])
    rag_mode = state.get("rag_mode", "unknown")
    thread_id = state.get("thread_id", "default")
    
    print("📝 生成最终答案...")
    
    if not context:
        final_answer = "未找到相关种植知识，请调整查询关键词后重试。"
    else:
        context_text = "\n".join(context)
        user_prompt = f"""请基于以下上下文，回答用户问题：
上下文：
{context_text}

用户问题：{query}

要求：
1. 回答准确、简洁，符合食用菌种植专业知识
2. 仅返回回答内容，无需额外解释
"""
        try:
            engine = get_engine()
            response_text = engine.llm.chat(
                prompt=user_prompt,
                history=[],
                content=context_text
            )
            final_answer = response_text.strip()
        except Exception as e:
            final_answer = f"无法生成答案，错误原因：{str(e)}"
    
    return {
        "final_answer": final_answer,
        "rag_mode": rag_mode,
        "thread_id": thread_id,
        "step": "answer_generated",
        "messages": [AIMessage(content="✅ 最终答案生成完成")]
    }

def finalize_node(state: RAGState) -> dict:
    """节点5：整理最终结果"""
    rag_mode = state.get("rag_mode", "unknown")
    thread_id = state.get("thread_id", "default")
    
    return {
        "step": "completed",
        "rag_mode": rag_mode,
        "thread_id": thread_id,
        "messages": [AIMessage(content="RAG引擎工作流完成")]
    }

def error_handler_node(state: RAGState) -> dict:
    """错误处理节点"""
    error = state.get("error", "未知错误")
    rag_mode = state.get("rag_mode", "unknown")
    thread_id = state.get("thread_id", "default")
    
    return {
        "final_answer": f"处理失败：{error}",
        "rag_mode": rag_mode,
        "thread_id": thread_id,
        "step": "error_handled",
        "messages": [AIMessage(content=f"❌ 处理出错：{error}")]
    }

def should_continue_retrieval(state: RAGState) -> str:
    """条件边：判断是否继续检索"""
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 1)
    need_retrieve = state.get("need_retrieve", False)
    error = state.get("error")
    
    if error:
        return "error_handler"
    elif need_retrieve and current_round < max_rounds:
        return "retrieve"
    else:
        return "generate_answer"

def should_reason_after_retrieve(state: RAGState) -> str:
    """条件边：检索后是否进行推理"""
    if state.get("error"):
        return "error_handler"
    return "reasoning"

def create_rag_engine_agent():
    """创建工作流图"""
    workflow = StateGraph(RAGState)
    
    # 添加节点
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("generate_answer", generate_final_answer_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # 从开始到初始化
    workflow.add_edge(START, "initialize")
    
    # 初始化后开始检索
    workflow.add_conditional_edges(
        "initialize",
        lambda state: "retrieve" if not state.get("error") else "error_handler",
        {
            "retrieve": "retrieve",
            "error_handler": "error_handler"
        }
    )
    
    # 检索后进行推理
    workflow.add_conditional_edges(
        "retrieve",
        should_reason_after_retrieve,
        {
            "reasoning": "reasoning",
            "error_handler": "error_handler"
        }
    )
    
    # 推理后决定是否继续检索
    workflow.add_conditional_edges(
        "reasoning",
        should_continue_retrieval,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
            "error_handler": "error_handler"
        }
    )
    
    # 生成答案后结束
    workflow.add_edge("generate_answer", "finalize")
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def run_rag_engine(query: str, planner_data: Dict) -> Dict:
    """运行RAG引擎的主函数"""
    app = create_rag_engine_agent()
    
    # 生成thread_id
    thread_id = f"rag-{hash(query)}-{hash(str(planner_data))}"
    
    inputs = {
        "messages": [HumanMessage(content=f"开始处理查询：{query}")],
        "query": query,
        "planner_data": planner_data,
        "thread_id": thread_id,
        "step": "start"
    }
    
    try:
        result = app.invoke(
            inputs,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # 构建返回结果
        return {
            "success": not result.get("error"),
            "retrieval_results": result.get("retrieval_results", []),
            "final_answer": result.get("final_answer", ""),
            "retrieve_rounds": result.get("current_round", 0),
            "rag_mode": result.get("rag_mode", "unknown"),
            "complexity": result.get("planner_data", {}).get("complexity", "simple"),
            "step": result.get("step", "completed"),
            "error": result.get("error")
        }
    except Exception as e:
        # 从planner_data中提取rag_mode作为备用
        rag_mode = "unknown"
        if planner_data and isinstance(planner_data, dict):
            planner_core = planner_data.get("data", planner_data)
            rag_mode = planner_core.get("mode", "unknown")
        
        return {
            "success": False,
            "final_answer": f"工作流执行失败：{str(e)}",
            "retrieve_rounds": 0,
            "rag_mode": rag_mode,
            "error": str(e)
        }

if __name__ == "__main__":
    # 测试代码
    print("===== 测试：复杂RAG问题 =====")
    test_planner = {
        "success": True,
        "data": {
            "retrievers": [
            "vector",
            "bm25"
            ],
            "top_k": 7,
            "rerank": True,
            "multi_round": True,
            "expand_keywords": [
            "北京",
            "香菇",
            "种植",
            "温度",
            "湿度",
            "适宜"
            ],
            "mode": "complex",
            "sub_tasks": [
            "北京香菇种植适宜温度",
            "北京香菇种植适宜湿度"
            ]
        },
        "step": "completed",
        "error": None
    }
    
    result = run_rag_engine("北京种植香菇需要控制哪些温度和湿度条件？", test_planner)
    print(json.dumps({
        "success": result["success"],
        "rag_mode": result.get("rag_mode", "unknown"),
        "retrieve_rounds": result.get("retrieve_rounds", 0),
        "final_answer": result.get("final_answer", ""),
        "error": result.get("error")
    }, ensure_ascii=False, indent=2))
