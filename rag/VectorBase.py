#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   VectorBase.py
@Time    :   2026/03/10 23:12:50
@Author  :   lemonnmin
@Version :   2.0
@Desc    :   基于FAISS的轻量级向量数据库实现（替换Chroma，解决hnsw索引错误）
"""

import os
import uuid
import sys
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import faiss
from rag.Embeddings import BaseEmbeddings, OpenAIEmbedding

# ========== 关键修复1：设置全局编码 + 兼容中文路径 ==========
import locale
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
# 强制使用绝对路径，避免相对路径混乱
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "faiss_log.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局单例FAISS存储
_faiss_index = None
_faiss_documents = None
_faiss_metadata = None

DEFAULT_INDEX_TYPE = "FLAT"  # 改用FLAT索引，避免IVF_FLAT训练步骤
N_LIST = 100

def init_faiss_index(dimension: int, index_type: str = DEFAULT_INDEX_TYPE):
    """初始化FAISS索引（单例模式）"""
    global _faiss_index
    if _faiss_index is None:
        if index_type == "IVF_FLAT":
            quantizer = faiss.IndexFlatL2(dimension)
            _faiss_index = faiss.IndexIVFFlat(quantizer, dimension, N_LIST, faiss.METRIC_L2)
        elif index_type == "FLAT":
            _faiss_index = faiss.IndexFlatL2(dimension)
        elif index_type == "HNSW":
            _faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"不支持的索引类型：{index_type}")
    return _faiss_index

class VectorStore:
    def __init__(self, document: List[str] = None, collection_name: str = "rag_docs", storage_path: str = None) -> None:
        """
        初始化FAISS向量数据库（修复目录创建和中文路径问题）
        """
        # ========== storage路径 ==========
        self.storage_path = storage_path or os.path.join(BASE_DIR, "storage")
        # 强制创建目录（无论是否存在）
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 集合相关配置（使用绝对路径）
        self.collection_name = collection_name
        self.index_file = os.path.join(self.storage_path, f"{collection_name}.index")
        self.doc_file = os.path.join(self.storage_path, f"{collection_name}_docs.pkl")
        
        # 初始化属性
        self.document = document if document is not None else []
        self.vectors = np.array([])  # 改为numpy数组，避免后续转换错误
        self.index = None
        self.bm25 = None
        self.bm25_corpus = None
        self.reranker = None
        
        # 尝试加载已有索引（修复后）
        if os.path.exists(self.index_file) and os.path.exists(self.doc_file):
            try:
                self.load_vector(path=self.storage_path, collection_name=collection_name)
            except Exception as e:
                logger.warning(f"加载已有索引失败：{e}，将创建新索引")

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """生成文档向量并入库（修复向量转换和目录写入）"""
        if not self.document:
            logger.warning("文档列表为空，跳过向量生成")
            return []
        
        # 文档预处理
        processed_docs = []
        for doc in self.document:
            doc = doc.strip().replace("\n", " ").replace("　", " ").replace("\r", "")
            if doc:
                processed_docs.append(doc)
        self.document = processed_docs
        total_docs = len(processed_docs)
        
        if total_docs == 0:
            logger.warning("预处理后无有效文档")
            return []
        
        # 批量生成向量
        if hasattr(EmbeddingModel, "get_embeddings_batch"):
            self.vectors = EmbeddingModel.get_embeddings_batch(self.document)
        else:
            self.vectors = []
            logger.info(f"开始生成 {total_docs} 个文档的向量...")
            for idx, doc in enumerate(self.document):
                try:
                    vec = EmbeddingModel.get_embedding(doc)
                    if sum(np.abs(vec)) < 1e-6:
                        logger.warning(f"第{idx+1}个文档向量为全0，跳过入库：{doc[:20]}...")
                        continue
                    self.vectors.append(vec)
                    if (idx+1) % 5 == 0:
                        logger.info(f"已处理 {idx+1}/{total_docs} 个文档")
                except Exception as e:
                    logger.error(f"生成第{idx+1}个文档向量失败：{e}，跳过")
        
        # ========== 确保向量是float32格式（FAISS要求） ==========
        if self.vectors:
            self.vectors = np.array(self.vectors).astype('float32')
            dimension = self.vectors.shape[1]
            logger.info(f"生成向量维度：{dimension}，数量：{len(self.vectors)}")
            
            # 初始化索引
            self.index = init_faiss_index(dimension)
            
            # 训练索引
            if not self.index.is_trained and isinstance(self.index, faiss.IndexIVFFlat):
                self.index.train(self.vectors)
            
            # 添加向量
            self.index.add(self.vectors)
            logger.info(f"成功入库 {len(self.vectors)} 个有效文档")
            
            # 立即持久化（避免索引未写入）
            self.persist(path=self.storage_path)
        else:
            logger.warning("无有效向量，跳过入库")
        
        return self.vectors.tolist() if len(self.vectors) > 0 else []

    def persist(self, path: str = 'storage'):
        """持久化FAISS索引（强制创建目录+绝对路径）"""
        if self.index is None or len(self.document) == 0:
            logger.warning("无数据可持久化")
            return
        
        # ========== 再次确认目录存在 ==========
        self.storage_path = os.path.abspath(path)
        os.makedirs(self.storage_path, exist_ok=True)  # 强制创建
        self.index_file = os.path.join(self.storage_path, f"{self.collection_name}.index")
        self.doc_file = os.path.join(self.storage_path, f"{self.collection_name}_docs.pkl")
        
        try:
            # 写入FAISS索引（二进制模式）
            faiss.write_index(self.index, self.index_file)
            # 写入文档（pickle序列化）
            with open(self.doc_file, 'wb') as f:
                pickle.dump({
                    'documents': self.document,
                    'vectors': self.vectors.tolist()
                }, f)
            logger.info(f"FAISS索引已保存到：{self.index_file}")
            logger.info(f"文档已保存到：{self.doc_file}")
        except PermissionError:
            raise ValueError(f"无写入权限：{self.storage_path}，请以管理员身份运行")
        except Exception as e:
            raise ValueError(f"持久化失败：{e}")

    def load_vector(self, path: str = 'storage', collection_name: str = "rag_docs"):
        """
        加载本地FAISS向量库（兼容原有接口）
        :param path: FAISS持久化目录
        :param collection_name: 集合名称（对应索引文件前缀）
        """
        self.storage_path = os.path.abspath(path)
        self.collection_name = collection_name
        self.index_file = os.path.join(self.storage_path, f"{collection_name}.index")
        self.doc_file = os.path.join(self.storage_path, f"{collection_name}_docs.pkl")
        
        # 检查文件是否存在
        if not os.path.exists(self.index_file):
            raise ValueError(f"FAISS索引文件不存在：{self.index_file}")
        if not os.path.exists(self.doc_file):
            raise ValueError(f"文档文件不存在：{self.doc_file}")
        
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(self.index_file)
            # 加载文档和向量
            with open(self.doc_file, 'rb') as f:
                data = pickle.load(f)
            self.document = data['documents']
            self.vectors = np.array(data['vectors']).astype('float32')
            
            # 校验加载结果
            doc_count = len(self.document)
            vec_count = len(self.vectors)
            logger.info(f"成功加载FAISS向量库：")
            logger.info(f"  - 文档数量：{doc_count}")
            logger.info(f"  - 向量数量：{vec_count}")
            logger.info(f"  - 向量维度：{self.vectors.shape[1] if vec_count > 0 else 0}")
            
            if doc_count == 0:
                logger.warning("加载的向量库为空！")
                
        except Exception as e:
            raise ValueError(f"加载FAISS向量库失败：{e}（请确认目录和集合名称正确）")

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

    # 1. 新增：关键词扩展方法
    def expand_query(self, query: str, expand_keywords: Optional[List[str]] = None) -> str:
        """扩展查询词（针对复杂问题）"""
        if not expand_keywords or len(expand_keywords) == 0:
            return query
        # 拼接扩展关键词，保留原查询语义
        expanded_query = f"{query} {' '.join(expand_keywords)}"
        logger.info(f"扩展后查询：{expanded_query}")
        return expanded_query

    # 2. 改造原 query 方法，支持多模式检索
    def query(
        self, 
        query: str, 
        EmbeddingModel: BaseEmbeddings, 
        k: int = 5,
        retrievers: List[str] = ["hybrid"],  # 新增：指定检索类型
        rerank: bool = True,  # 新增：是否重排序
        expand_keywords: List[str] = None,  # 新增：关键词扩展
        history_queries: List[str] = None  # 新增：多轮检索的历史上下文
    ) -> List[str]:
        """
        增强版检索：支持多模式/关键词扩展/多轮上下文
        :param query: 原始查询
        :param EmbeddingModel: 嵌入模型
        :param k: 返回结果数
        :param retrievers: 检索类型：vector/bm25/hybrid
        :param rerank: 是否重排序
        :param expand_keywords: 关键词扩展列表
        :param history_queries: 多轮检索的历史查询列表
        :return: 检索结果列表
        """
        if k <= 0:
            raise ValueError("k值必须大于0")

        # ========= 0 预处理：多轮上下文 + 关键词扩展 =========
        final_query = query
        # 多轮检索：拼接历史查询
        if history_queries and len(history_queries) > 0:
            final_query = f"{' '.join(history_queries)} {query}"
            logger.info(f"多轮检索拼接后查询：{final_query}")
        # 关键词扩展
        if expand_keywords:
            final_query = self.expand_query(final_query, expand_keywords)
        
        # 基础预处理
        final_query = final_query.strip().replace("\n", " ").replace("　", " ").replace("\r", "")
        if not final_query:
            logger.warning("查询文本为空！")
            return []

        # ========= 1 生成查询向量 =========
        query_vector = EmbeddingModel.get_embedding(final_query)
        if sum(np.abs(query_vector)) < 1e-6:
            logger.warning("查询向量为全0，无法检索！")
            return []
        query_vec = np.array([query_vector]).astype('float32')

        # ========= 2 按检索类型获取候选文档 =========
        vector_docs = []
        bm25_docs = []

        # 向量检索（vector）
        if "vector" in retrievers or "hybrid" in retrievers:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = 10
            distances, indices = self.index.search(query_vec, k * 3)
            for idx in indices[0]:
                if 0 <= idx < len(self.document):
                    vector_docs.append(self.document[idx])
            logger.info(f"向量检索返回 {len(vector_docs)} 个候选")

        # BM25检索（bm25）
        if "bm25" in retrievers or "hybrid" in retrievers:
            try:
                if not hasattr(self, "bm25") or self.bm25_corpus is None:
                    self.bm25_corpus = self.document
                    tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                bm25_docs = self.bm25.get_top_n(final_query.split(), self.bm25_corpus, n=k * 2)
                logger.info(f"BM25检索返回 {len(bm25_docs)} 个候选")
            except Exception as e:
                logger.warning(f"BM25检索失败: {e}")
                bm25_docs = []

        # ========= 3 合并候选 =========
        if retrievers == ["vector"]:
            combined_docs = vector_docs
        elif retrievers == ["bm25"]:
            combined_docs = bm25_docs
        else:  # hybrid 或 多类型
            combined_docs = list(set(vector_docs + bm25_docs))

        if not combined_docs:
            logger.warning("无检索结果")
            return []
        logger.info(f"合并候选文档数: {len(combined_docs)}")

        # ========= 4 重排序（可选） =========
        if rerank:
            try:
                if not hasattr(self, "reranker") or self.reranker is None:
                    logger.info("加载BAAI/bge-reranker-base重排序模型...")
                    cache_dir = os.path.join(os.path.dirname(__file__), "models")
                    os.makedirs(cache_dir, exist_ok=True)
                    self.reranker = CrossEncoder(
                        "BAAI/bge-reranker-base",
                        cache_dir=cache_dir,
                        device="cpu"
                    )
                # 重排序：使用最终查询（含扩展/多轮上下文）
                pairs = [(final_query, doc) for doc in combined_docs]
                scores = self.reranker.predict(pairs)
                ranked_idx = np.argsort(scores)[::-1]
                final_docs = [combined_docs[i] for i in ranked_idx[:k]]
                logger.info(f"重排序后返回 {len(final_docs)} 个结果")
                return final_docs
            except Exception as e:
                logger.error(f"重排序失败：{e}", exc_info=True)
                logger.warning("降级返回未排序的候选结果")
                return combined_docs[:k]
        else:
            # 不重排序：直接返回前k个
            return combined_docs[:k]

    # 新增实用方法
    def clear_collection(self):
        """清空当前集合的所有数据"""
        if self.index is not None:
            self.index.reset()
        self.document = []
        self.vectors = np.array([])
        # 删除持久化文件
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.doc_file):
            os.remove(self.doc_file)
        logger.info("已清空FAISS向量库所有数据")

    def get_collection_stats(self):
        """获取向量库统计信息"""
        stats = {
            "文档数量": len(self.document),
            "向量数量": len(self.vectors) if hasattr(self.vectors, '__len__') else 0,
            "向量维度": self.vectors.shape[1] if len(self.vectors) > 0 else 0,
            "集合名称": self.collection_name,
            "持久化目录": self.storage_path,
            "索引类型": type(self.index).__name__ if self.index is not None else "未初始化"
        }
        logger.info("FAISS向量库统计信息：")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
        return stats

# 测试代码（可选）
if __name__ == "__main__":
    # 测试FAISS向量库
    test_docs = [
        "常见食用菌包括香菇、金针菇、杏鲍菇等",
        "金针菇适合在低温环境下种植，温度10-15℃最佳",
        "香菇的种植需要木屑、麸皮等培养基",
        "杏鲍菇的生长周期约30天，适合在通风良好的环境中种植"
    ]
    
    # 初始化向量库
    vs = VectorStore(document=test_docs, collection_name="test_docs")
    
    # 初始化Embedding模型（请替换为你的实际模型）
    embedding_model = OpenAIEmbedding()  # 或其他BaseEmbeddings子类
    
    # 生成并入库向量
    vectors = vs.get_vector(embedding_model)
    logger.info(f"生成向量数：{len(vectors)}")
    
    # 持久化
    vs.persist()
    
    # 检索测试
    results = vs.query("金针菇种植条件", embedding_model, k=2)
    logger.info("检索结果：")
    for i, doc in enumerate(results, 1):
        logger.info(f"  {i}. {doc}")
    
    # 加载测试
    vs2 = VectorStore()
    vs2.load_vector(collection_name="test_docs")
    vs2.get_collection_stats()