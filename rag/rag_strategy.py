#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   rag_strategy.py
@Time    :   2026/03/12 17:59:13
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RetrievalPlan:
    """检索策略配置类"""
    retrievers: List[str]  # 检索类型：["vector"] / ["vector", "bm25"] / ["hybrid"]
    top_k: int  # 返回结果数
    rerank: bool  # 是否重排序
    multi_round: bool  # 是否多轮检索
    expand_keywords: Optional[List[str]] = None  # 关键词扩展列表

# 预定义检索策略
RETRIEVAL_STRATEGIES = {
    ("rag_search", "simple"): RetrievalPlan(
        retrievers=["vector"],
        top_k=3,
        rerank=True,
        multi_round=False
    ),
    ("rag_search", "complex"): RetrievalPlan(
        retrievers=["vector", "bm25"],
        top_k=5,
        rerank=True,
        multi_round=True,
        expand_keywords=["种植技术", "环境控制"]
    ),
    ("rag_search", "multi_turn"): RetrievalPlan(
        retrievers=["hybrid"],
        top_k=4,
        rerank=True,
        multi_round=True
    )
}

def get_retrieval_strategy(task: str, mode: str) -> RetrievalPlan:
    """获取指定任务+模式的检索策略"""
    key = (task, mode)
    if key not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"无匹配的检索策略：{key}，可选策略：{list(RETRIEVAL_STRATEGIES.keys())}")
    return RETRIEVAL_STRATEGIES[key]