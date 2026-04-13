#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   rag_search.py
@Time    :   2026/03/10 22:50:55
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   rag_search 专门调用Chroma，Embedding，RAG 检索
"""
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List
from rag.VectorBase import VectorStore
from rag.Embeddings import OpenAIEmbedding
from rag.LLM import OpenAIChat
from rag.rag_strategy import get_retrieval_strategy, RetrievalPlan
from tools.base_tool import BaseTool

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

class RagSearchTool(BaseTool):
    name = "rag_search"
    description = "用于查询食用菌种植知识库，支持simple/complex/multi_turn三种检索模式"

    def __init__(self, mode: str = "simple"):
        """
        初始化检索工具
        :param mode: 检索模式：simple/complex/multi_turn
        """
        # 获取对应模式的检索策略
        load_env_from_root()
        self.strategy: RetrievalPlan = get_retrieval_strategy("rag_search", mode)
        self.embedding = OpenAIEmbedding()
        self.vector_db = VectorStore()
        self.llm = OpenAIChat()
        # 多轮检索的历史上下文
        self.history_queries = []

    def retrieve(self, query: str) -> List[str]:
        """
        按策略执行检索
        :param query: 单次查询文本
        :return: 检索结果列表
        """
        docs = self.vector_db.query(
            query=query,
            EmbeddingModel=self.embedding,
            k=self.strategy.top_k,
            retrievers=self.strategy.retrievers,
            rerank=self.strategy.rerank,
            expand_keywords=self.strategy.expand_keywords,
            history_queries=self.history_queries if self.strategy.multi_round else None
        )
        # 多轮检索：记录历史查询
        if self.strategy.multi_round:
            self.history_queries.append(query)
            # 限制历史长度，避免上下文过长
            if len(self.history_queries) > 5:
                self.history_queries = self.history_queries[-5:]
        return docs

    def run(self, query: str) -> str:
        """执行检索并格式化结果"""
        try:
            docs = self.retrieve(query)
            results = []
            for i, doc in enumerate(docs, start=1):
                results.append({
                    "rank": i,
                    "content": doc,
                    "score": None,
                    "source": "rag_db"
                })
            return self.format_result(query, results)
        except Exception as e:
            return self.format_error(query, e)

    def clear_history(self):
        """清空多轮检索的历史上下文"""
        self.history_queries = []
        self.logger.info("已清空多轮检索历史")

    # 补充格式化方法（如果 BaseTool 未实现）
    def format_result(self, query: str, results: List[dict]) -> str:
        """格式化成功结果"""
        res_str = f"✅ 查询「{query}」的检索结果：\n"
        for item in results:
            res_str += f"  排名{item['rank']}：{item['content']}\n"
        return res_str.strip()

    def format_error(self, query: str, e: Exception) -> str:
        """格式化错误结果"""
        return f"❌ 查询「{query}」失败：{str(e)}"

if __name__ == "__main__":
    # 测试不同检索模式
    # 1. 简单模式（仅向量检索，top_k=3）
    print("===== 简单模式 =====")
    simple_tool = RagSearchTool(mode="simple")
    print(simple_tool.run("香菇适宜的生长温度是多少？"))

    # # 2. 复杂模式（混合检索+关键词扩展+多轮，top_k=5）
    # print("\n===== 复杂模式 =====")
    # complex_tool = RagSearchTool(mode="complex")
    # print(complex_tool.run("金针菇种植需要注意什么？"))
    # # 多轮查询（复用上下文）
    # print(complex_tool.run("它的湿度要求是多少？"))

    # # 3. 多轮模式（混合检索+多轮，top_k=4）
    # print("\n===== 多轮模式 =====")
    # multi_tool = RagSearchTool(mode="multi_turn")
    # print(multi_tool.run("杏鲍菇的生长周期是多久？"))
    # print(multi_tool.run("如何缩短这个周期？"))