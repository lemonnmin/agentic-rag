#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   Embeddings.py
@Time    :   2026/03/10 23:13:57
@Author  :   lemonnmin
@Version :   1.2
@Desc    :   支持API/本地模式的嵌入模型，含重试/日志/批量处理
"""


import os
import logging
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI
try:
    # 新版本 OpenAI（v1.x+）
    from openai.exceptions import APIError, APIConnectionError, RateLimitError
except ImportError:
    # 旧版本 OpenAI（v0.x）
    from openai import APIError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 加载.env文件
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# ---------------------- 全局日志配置 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("embedding.log"), logging.StreamHandler()]
)

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        初始化嵌入基类
        Args:
            path (str): 模型或数据的路径
            is_api (bool): 是否使用API方式。True表示使用在线API服务，False表示使用本地模型
        """
        self.path = path
        self.is_api = is_api
        # 初始化日志（关键修复：给基类添加logger属性）
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_embedding(self, text: str, model: str = "") -> List[float]:
        """
        获取文本的嵌入向量表示（统一参数格式，添加默认model）
        Args:
            text (str): 输入文本
            model (str): 使用的模型名称
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        Args:
            vector1 (List[float]): 第一个向量
            vector2 (List[float]): 第二个向量
        Returns:
            float: 两个向量的余弦相似度，范围在[-1,1]之间
        """
        # 将输入列表转换为numpy数组，并指定数据类型为float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # 检查向量中是否包含无穷大或NaN值
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        # 计算向量的范数（长度）
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算分母（两个向量范数的乘积）
        magnitude = norm_v1 * norm_v2
        # 处理分母为0的特殊情况
        if magnitude == 0:
            return 0.0
            
        # 返回余弦相似度
        return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    优化版：支持API调用（硅基流动bge-m3）+ 本地bge模型，添加重试/异常处理
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        self.embedding_model = None  # 本地模型实例
        
        # 1. API模式初始化（硅基流动）
        if self.is_api:
            # 校验环境变量
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL")
            if not self.api_key or not self.base_url:
                raise ValueError("API模式下必须设置 OPENAI_API_KEY 和 OPENAI_BASE_URL 环境变量！")
            
            # 初始化OpenAI客户端
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.logger.info("硅基流动API客户端初始化成功")
        
        # 2. 本地模式初始化（加载bge-m3本地模型）
        else:
            try:
                from sentence_transformers import SentenceTransformer
                # 加载本地模型（path为空则自动下载到缓存）
                self.embedding_model = SentenceTransformer(
                    self.path if self.path else "BAAI/bge-m3"
                )
                self.logger.info("本地bge-m3模型加载成功")
            except ImportError:
                raise ImportError("本地模式需安装sentence-transformers：pip install sentence-transformers")
            except Exception as e:
                raise RuntimeError(f"本地模型加载失败：{e}")

    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 等待时间：2s→4s→8s
        retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError)),
        reraise=True  # 重试失败后抛出原异常
    )
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        生成嵌入向量：API模式调用硅基流动bge-m3，本地模式用sentence-transformers
        :param text: 输入文本
        :param model: 嵌入模型名称（API模式有效）
        :return: 浮点型向量列表
        """
        # 通用文本预处理（提升嵌入效果）
        text = self._preprocess_text(text)
        
        # 空文本返回零向量（关键修复：避免API/本地模型报错）
        if not text:
            # bge-m3向量维度是1024，返回等长零向量
            zero_vec = [0.0] * 1024
            self.logger.warning("输入文本为空，返回1024维零向量")
            return zero_vec
        
        # 1. API模式（硅基流动）
        if self.is_api:
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=model,
                    encoding_format="float"  # 指定返回float（默认是base64，需转换）
                )
                embedding = response.data[0].embedding
                self.logger.debug(f"API模式生成向量长度：{len(embedding)}")
                return embedding
            except Exception as e:
                self.logger.error(f"API调用失败：{e}", exc_info=True)
                raise  # 抛出异常供上层处理
        
        # 2. 本地模式
        else:
            try:
                # 生成向量（bge-m3建议添加检索指令前缀，提升效果）
                text = f"为句子生成嵌入以用于检索：{text}"
                embedding = self.embedding_model.encode(
                    text,
                    normalize_embeddings=True,  # 归一化向量（提升余弦相似度计算效果）
                    convert_to_numpy=True
                ).tolist()
                self.logger.debug(f"本地模式生成向量长度：{len(embedding)}")
                return embedding
            except Exception as e:
                self.logger.error(f"本地模型生成向量失败：{e}", exc_info=True)
                raise

    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理：清洗无关字符，提升嵌入质量
        """
        if not isinstance(text, str) or text.strip() == "":
            self.logger.warning("输入文本为空，返回零向量")
            return ""  # 空文本后续返回零向量
        
        # 清洗规则：去除多余换行/空格、全角转半角、去除特殊字符
        text = text.replace("\n", " ").replace("\r", " ").strip()
        text = text.replace("　", " ")  # 全角空格转半角
        text = "".join([c for c in text if c.isprintable()])  # 保留可打印字符
        return text

    def get_embeddings_batch(self, texts: List[str], model: str = "BAAI/bge-m3", batch_size: int = 32) -> List[List[float]]:
        """
        批量生成嵌入向量（提升效率，避免单条调用）
        :param texts: 文本列表
        :param batch_size: 批次大小
        :return: 向量列表
        """
        embeddings = []
        # 分批次处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # 预处理批次文本
            batch_texts = [self._preprocess_text(t) for t in batch_texts]
            
            if self.is_api:
                # API批量调用
                response = self.client.embeddings.create(input=batch_texts, model=model)
                batch_emb = [item.embedding for item in response.data]
            else:
                # 本地批量调用
                batch_texts = [f"为句子生成嵌入以用于检索：{t}" for t in batch_texts]
                batch_emb = self.embedding_model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).tolist()
            
            embeddings.extend(batch_emb)
        return embeddings