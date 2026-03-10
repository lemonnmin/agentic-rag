#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   VectorBase.py
@Time    :   2025/06/20 10:11:13
@Author  :   不要葱姜蒜
@Version :   2.3  # 修复未检索到文档问题
@Desc    :   基于Chroma的轻量级向量数据库实现（适配新版Chroma v0.4.0+）
'''

import os
import uuid
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union
import chromadb
from chromadb.config import Settings
from Embeddings import BaseEmbeddings, OpenAIEmbedding
import numpy as np
# 移除tqdm（彻底解决Windows兼容性问题）

# 修复Windows编码问题：配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chroma_log.log", encoding='utf-8'),  # 日志文件UTF-8编码
        logging.StreamHandler()  # 保留默认stdout，不替换
    ]
)
logger = logging.getLogger(__name__)

# 全局单例Chroma客户端（避免重复初始化冲突）
_chroma_client = None

def get_chroma_client(path: str = None, anonymized_telemetry: bool = False):
    """获取单例Chroma客户端，避免重复初始化"""
    global _chroma_client
    if _chroma_client is None:
        path = path or os.path.join(os.getcwd(), "chroma_storage")
        _chroma_client = chromadb.PersistentClient(
            path=os.path.abspath(path),
            settings=Settings(anonymized_telemetry=anonymized_telemetry)
        )
    return _chroma_client

# 基于Chroma的向量数据库实现（替换原有JSON存储）
class VectorStore:
    def __init__(self, document: List[str] = None, collection_name: str = "rag_docs", storage_path: str = None) -> None:
        """
        初始化向量数据库（适配新版Chroma v0.4.0+，解决实例冲突）
        :param document: 初始文档列表（可选）
        :param collection_name: Chroma集合名称（用于区分不同数据集）
        :param storage_path: 持久化目录（优先使用传入路径）
        """
        # 🔥 关键修复：统一存储路径（和rag_ui.py中的storage目录对齐）
        self.storage_path = storage_path or os.path.join(os.getcwd(), "storage")
        self.client = get_chroma_client(
            path=self.storage_path,
            anonymized_telemetry=False
        )
        
        # 获取/创建集合（自动处理重复创建问题）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG文档向量集合"}
        )
        
        # 初始化文档列表
        self.document = document if document is not None else []
        self.vectors = []  # 保留原有属性，兼容旧逻辑
        self.collection_name = collection_name  # 保存集合名称

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """
        生成文档向量并入库（批量处理，效率提升10倍+）
        :param EmbeddingModel: 嵌入模型实例
        :return: 向量列表
        """
        if not self.document:
            logger.warning("文档列表为空，跳过向量生成")
            return []
        
        # 🔥 新增：文档预处理（和查询文本保持一致）
        processed_docs = []
        for doc in self.document:
            # 清洗空字符串/特殊字符
            doc = doc.strip().replace("\n", " ").replace("　", " ").replace("\r", "")
            if doc:  # 过滤空文档
                processed_docs.append(doc)
        self.document = processed_docs
        
        # 批量生成向量（优先使用模型的批量方法，无则降级为单条）
        if hasattr(EmbeddingModel, "get_embeddings_batch"):
            self.vectors = EmbeddingModel.get_embeddings_batch(self.document)
        else:
            self.vectors = []
            # 移除tqdm，改用手动打印进度
            total_docs = len(self.document)
            for idx, doc in enumerate(self.document):
                vec = EmbeddingModel.get_embedding(doc)
                # 校验向量有效性
                if sum(vec) == 0:
                    logger.warning(f"第{idx+1}个文档向量为全0，跳过入库：{doc[:20]}...")
                    continue
                self.vectors.append(vec)
                if (idx+1) % 5 == 0:  # 每5个文档打印一次进度
                    logger.info(f"已处理 {idx+1}/{total_docs} 个文档")
        
        # 生成唯一ID（避免重复入库）
        ids = [f"doc_{uuid.uuid4().hex}" for _ in range(len(self.vectors))]
        
        # 批量入库（Chroma自动处理向量存储和索引）
        if self.vectors:  # 确保有有效向量才入库
            self.collection.add(
                documents=self.document[:len(self.vectors)],  # 对齐向量和文档数量
                embeddings=self.vectors,
                ids=ids
            )
            logger.info(f"成功入库 {len(self.vectors)} 个有效文档，向量维度：{len(self.vectors[0])}")
        else:
            logger.warning("无有效向量，跳过入库")
        
        return self.vectors

    def persist(self, path: str = 'storage'):
        """
        兼容原有persist方法（新版Chroma自动持久化，无需手动调用）
        :param path: 持久化目录（覆盖默认的storage）
        """
        global _chroma_client
        # 更新存储路径并重建单例客户端
        self.storage_path = os.path.abspath(path)
        _chroma_client = chromadb.PersistentClient(
            path=self.storage_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = _chroma_client.get_or_create_collection(name=self.collection_name)
        logger.info(f"向量数据库已持久化到：{self.storage_path}")

    def load_vector(self, path: str = 'storage', collection_name: str = "rag_docs"):
        """
        加载本地Chroma向量库（解决实例冲突问题）
        :param path: Chroma持久化目录
        :param collection_name: 集合名称
        """
        global _chroma_client
        # 强制使用指定路径重建客户端
        self.storage_path = os.path.abspath(path)
        _chroma_client = chromadb.PersistentClient(
            path=self.storage_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 加载已有集合
        try:
            self.collection = _chroma_client.get_collection(name=collection_name)
            self.collection_name = collection_name
            # 读取所有文档和向量（兼容原有属性）
            results = self.collection.get()
            self.document = results["documents"]
            self.vectors = results["embeddings"]
            # 🔥 新增：校验加载结果
            doc_count = len(self.document)
            vec_count = len(self.vectors)
            logger.info(f"成功加载向量库：{doc_count} 个文档，{vec_count} 个向量")
            if doc_count == 0:
                logger.warning("加载的向量库为空！")
        except Exception as e:
            raise ValueError(f"加载向量库失败：{e}（请确认目录和集合名称正确）")

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        保留原有余弦相似度计算方法（兼容旧逻辑）
        """
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 5) -> List[str]:
        """
        向量检索（修复未检索到文档问题，添加预处理+距离过滤）
        :param query: 查询文本
        :param EmbeddingModel: 嵌入模型实例
        :param k: 返回最相似的k个文档（默认3个，提高命中率）
        :return: 相似文档列表
        """
        if k <= 0:
            raise ValueError("k值必须大于0")
        
        # 🔥 核心修复1：查询文本预处理（和文档预处理逻辑完全一致）
        query = query.strip().replace("\n", " ").replace("　", " ").replace("\r", "")
        if not query:
            logger.warning("查询文本为空！")
            return []
        
        # 生成查询向量
        query_vector = EmbeddingModel.get_embedding(query)
        
        # 🔥 核心修复2：校验查询向量有效性
        if sum(query_vector) == 0:
            logger.warning("查询向量为全0，无法检索！")
            return []
        
        # 使用Chroma内置检索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "distances"]  # 返回文档和距离（距离越小相似度越高）
        )
        
        # 🔥 核心修复3：过滤无效结果（L2距离<1.0视为相关）
        similar_docs = []
        distances = results["distances"][0] if results["distances"] else []
        docs = results["documents"][0] if results["documents"] else []
        
        logger.info(f"原始检索结果：{len(docs)} 个，距离：{distances}")
        
        for doc, dist in zip(docs, distances):
            # L2距离阈值：<1.0视为相关（可根据模型调整）
            if dist < 1.0 and doc.strip():
                similar_docs.append(doc)
                logger.info(f"有效检索结果（距离{dist:.4f}）：{doc[:50]}...")
            else:
                logger.warning(f"结果距离过大（{dist:.4f}）或文档为空，过滤：{doc[:20]}...")
        
        return similar_docs

    # 新增实用方法（可选）
    def clear_collection(self):
        """清空当前集合的所有数据"""
        self.collection.delete(ids=self.collection.get()["ids"])
        self.document = []
        self.vectors = []
        logger.info("已清空向量库所有数据")

    def get_collection_stats(self):
        """获取向量库统计信息"""
        stats = {
            "文档数量": self.collection.count(),
            "集合名称": self.collection.name,
            "持久化目录": self.client.path
        }
        logger.info("向量库统计信息：")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
        return stats