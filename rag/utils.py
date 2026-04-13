#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/03/10 23:13:18
@Author  :   lemonnmin
@Version :   1.5  # 彻底修复tqdm锁参数错误
@Desc    :   文本读取与智能分块工具类
"""


import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup

# 全局初始化token编码器（避免重复初始化，提升效率）
enc = tiktoken.get_encoding("cl100k_base")

# 🔥 核心修复：彻底简化tqdm配置，移除所有锁参数（解决Windows下锁冲突）
# Windows下直接禁用tqdm的进度条显示（避免所有终端/I/O问题）
if sys.platform == "win32":
    tqdm_params = {
        'disable': True,  # 禁用进度条显示（关键修复）
        'file': sys.stdout
    }
else:
    tqdm_params = {
        'disable': False,
        'file': sys.stdout
    }


class ReadFiles:
    """
    文本文件读取与智能分块工具类
    支持：md/txt/pdf 文件读取，基于语义+Token的智能分块
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self) -> List[str]:
        """递归遍历指定目录，获取所有支持的文件列表（md/txt/pdf）"""
        file_list = []
        # 校验路径是否存在
        if not os.path.exists(self._path):
            print(f"警告：路径 {self._path} 不存在，返回空文件列表")
            return file_list
        
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                # 过滤支持的文件类型
                if filename.lower().endswith((".md", ".txt", ".pdf")):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """
        读取所有文件内容并进行智能分块
        :param max_token_len: 每个块的最大Token长度（含重叠）
        :param cover_content: 相邻块的重叠Token数
        :return: 分块后的文本列表
        """
        docs = []
        if not self.file_list:
            print("警告：未找到任何支持的文件，返回空列表")
            return docs
        
        # 🔥 核心修复：简化tqdm调用，移除所有锁参数
        # 方式1：禁用tqdm进度条（推荐，彻底解决Windows问题）
        for file in self.file_list:
            try:
                content = self.read_file_content(file)
                if content.strip():  # 跳过空内容
                    chunk_content = self.get_chunk(
                        content, max_token_len=max_token_len, cover_content=cover_content)
                    docs.extend(chunk_content)
                # 手动打印进度（替代tqdm）
                print(f"已处理文件：{file}")
            except Exception as e:
                print(f"错误：读取文件 {file} 失败 - {e}")
                continue
        
        # 方式2（备选）：如果仍需tqdm，使用极简配置
        # for file in tqdm(self.file_list, desc="读取并分块文件", disable=True):
        #     pass
        
        print(f"✅ 文件处理完成，共生成 {len(docs)} 个文本块")
        return docs

    @classmethod
    def get_chunk(cls, 
                  text: str, 
                  max_token_len: int = 600, 
                  cover_content: int = 150) -> List[str]:
        """
        智能文本分块：优先按段落/句子分割，再按Token长度调整，保留重叠窗口
        :param text: 原始文本
        :param max_token_len: 每个块的最大Token长度（含重叠内容）
        :param cover_content: 相邻块的重叠Token数（保证语义连贯）
        :return: 分块后的文本列表
        """
        # 前置校验：空文本直接返回空列表
        if not isinstance(text, str) or text.strip() == "":
            return []
        
        # 1. 参数合法性校验（避免无效参数）
        max_token_len = max(100, max_token_len)  # 最小块长度限制
        cover_content = min(cover_content, max_token_len // 2)  # 重叠不超过块长度的一半
        effective_len = max_token_len - cover_content  # 块的有效内容长度（扣除重叠）
        
        # 2. 文本预处理：保留段落结构，清洗无效字符
        text = cls._preprocess_text(text)
        # 优先按段落分割（最大程度保留语义）
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunk_text = []
        curr_chunk = ""
        curr_token_len = 0
        
        for para in paragraphs:
            # 计算当前段落的Token长度
            para_token_len = len(enc.encode(para))
            
            # 场景1：当前段落本身超过最大长度 → 拆分该段落（按句子再分）
            if para_token_len > max_token_len:
                # 先保存当前未完成的块
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ""
                    curr_token_len = 0
                
                # 按句子拆分超长段落（优先按中文标点分割）
                sentence_chunks = cls._split_long_text(para, effective_len, cover_content)
                chunk_text.extend(sentence_chunks)
                continue
            
            # 场景2：当前段落可加入当前块 → 直接添加
            if curr_token_len + para_token_len + 1 <= effective_len:  # +1 是换行符的Token
                if curr_chunk:
                    curr_chunk += "\n\n"  # 保留段落分隔符
                    curr_token_len += len(enc.encode("\n\n"))
                curr_chunk += para
                curr_token_len += para_token_len
            
            # 场景3：当前段落加入后超长度 → 保存当前块，新建块（带重叠）
            else:
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                
                # 新建块：添加重叠内容 + 当前段落
                curr_chunk = cls._add_cover_content(chunk_text, cover_content) + "\n\n" + para
                # 重新计算新块的Token长度（避免重叠内容超限制）
                curr_token_len = len(enc.encode(curr_chunk))
                # 兜底：如果重叠+段落仍超长度，强制截断（极端情况）
                if curr_token_len > max_token_len:
                    curr_chunk = cls._truncate_by_token(curr_chunk, max_token_len)
                    curr_token_len = len(enc.encode(curr_chunk))
        
        # 3. 处理最后一个未完成的块
        if curr_chunk.strip():
            chunk_text.append(curr_chunk.strip())
        
        # 4. 去重+清洗空块（避免重复/无效内容）
        chunk_text = [c for c in list(dict.fromkeys(chunk_text)) if c.strip()]
        
        return chunk_text
    
    @classmethod
    def _preprocess_text(cls, text: str) -> str:
        """文本预处理：保留语义的前提下清洗无效字符"""
        # 替换全角空格、连续换行
        text = text.replace("　", " ").replace("\r", "\n")
        # 合并连续空行（保留最多一个空行）
        lines = []
        last_empty = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                if not last_empty:
                    lines.append("")
                    last_empty = True
            else:
                lines.append(line)
                last_empty = False
        return "\n".join(lines)
    
    @classmethod
    def _split_long_text(cls, text: str, effective_len: int, cover_content: int) -> List[str]:
        """拆分超长文本：按句子分割，保证语义完整"""
        # 优先按中文标点分割句子（。！？；），其次按英文标点
        sentence_sep = re.compile(r'([。！？；])')
        sentences = sentence_sep.split(text)
        # 重组句子（把标点拼回句子末尾）
        sentences = [s1 + s2 for s1, s2 in zip(sentences[::2], sentences[1::2])] if len(sentences) > 1 else [text]
        
        sentence_chunks = []
        curr_sent_chunk = ""
        curr_sent_token = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_token = len(enc.encode(sent))
            
            # 单句超长 → 强制按Token分割（最后兜底）
            if sent_token > effective_len:
                if curr_sent_chunk:
                    sentence_chunks.append(curr_sent_chunk)
                # 按Token分割单句，保留重叠
                token_chunks = cls._split_by_token(sent, effective_len, cover_content)
                sentence_chunks.extend(token_chunks)
                curr_sent_chunk = ""
                curr_sent_token = 0
            # 单句可加入当前块
            elif curr_sent_token + sent_token + 1 <= effective_len:
                if curr_sent_chunk:
                    curr_sent_chunk += sent
                    curr_sent_token += sent_token
                else:
                    curr_sent_chunk = sent
                    curr_sent_token = sent_token
            # 新建块（带重叠）
            else:
                sentence_chunks.append(curr_sent_chunk)
                # 重叠内容 + 当前句子
                cover_part = cls._get_cover_content(sentence_chunks[-1], cover_content) if sentence_chunks else ""
                curr_sent_chunk = cover_part + sent
                # 修复：将 curr_chunk 改为 curr_sent_chunk
                curr_sent_token = len(enc.encode(curr_sent_chunk))
        
        if curr_sent_chunk.strip():
            sentence_chunks.append(curr_sent_chunk.strip())
        
        return sentence_chunks
    
    @classmethod
    def _split_by_token(cls, text: str, chunk_token_len: int, cover_content: int) -> List[str]:
        """纯Token分割（兜底方案）：仅在句子分割仍超长时使用"""
        tokens = enc.encode(text)
        chunks = []
        prev_chunk_tokens = []
        
        for i in range(0, len(tokens), chunk_token_len):
            # 计算当前块的Token范围
            start_idx = i
            end_idx = min(i + chunk_token_len, len(tokens))
            curr_chunk_tokens = tokens[start_idx:end_idx]
            
            # 添加重叠内容（除第一个块）
            if prev_chunk_tokens and cover_content > 0:
                cover_tokens = prev_chunk_tokens[-cover_content:]
                curr_chunk_tokens = cover_tokens + curr_chunk_tokens
            
            # 解码为文本并去重
            curr_chunk = enc.decode(curr_chunk_tokens).strip()
            if curr_chunk and curr_chunk not in chunks:
                chunks.append(curr_chunk)
            
            prev_chunk_tokens = curr_chunk_tokens
        
        return chunks
    
    @classmethod
    def _add_cover_content(cls, chunk_list: List[str], cover_content: int) -> str:
        """给新块添加重叠内容（保证语义连贯）"""
        if not chunk_list or cover_content <= 0:
            return ""
        # 取最后一个块的末尾N个Token作为重叠内容
        last_chunk = chunk_list[-1]
        return cls._get_cover_content(last_chunk, cover_content)
    
    @classmethod
    def _get_cover_content(cls, text: str, cover_token_num: int) -> str:
        """获取文本末尾指定Token数的内容（精准控制重叠长度）"""
        if cover_token_num <= 0 or not text:
            return ""
        # 编码为Token，取最后N个，再解码
        tokens = enc.encode(text)
        cover_tokens = tokens[-cover_token_num:] if len(tokens) >= cover_token_num else tokens
        return enc.decode(cover_tokens)
    
    @classmethod
    def _truncate_by_token(cls, text: str, max_token_len: int) -> str:
        """按Token长度截断文本（保留前N个Token）"""
        tokens = enc.encode(text)[:max_token_len]
        return enc.decode(tokens).strip()

    @classmethod
    def read_file_content(cls, file_path: str) -> str:
        """
        根据文件类型读取内容
        :param file_path: 文件路径
        :return: 纯文本内容
        """
        # 校验文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        # 按文件扩展名分发读取方法
        if file_path.lower().endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.lower().endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.lower().endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{os.path.splitext(file_path)[1]}")

    @classmethod
    def read_pdf(cls, file_path: str) -> str:
        """读取PDF文件并提取纯文本"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # 遍历所有页面提取文本
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()

    @classmethod
    def read_markdown(cls, file_path: str) -> str:
        """读取Markdown文件并提取纯文本（移除HTML标签和链接）"""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
        
        # 转换Markdown到HTML，再提取纯文本
        html_text = markdown.markdown(md_text)
        soup = BeautifulSoup(html_text, 'html.parser')
        plain_text = soup.get_text()
        
        # 移除网址链接、多余空格
        plain_text = re.sub(r'http\S+|https\S+', '', plain_text)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        
        return plain_text

    @classmethod
    def read_text(cls, file_path: str) -> str:
        """读取纯文本文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()


class Documents:
    """JSON格式文档读取工具类"""
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self) -> Union[Dict, List]:
        """读取JSON文件内容"""
        if not self.path or not os.path.exists(self.path):
            raise ValueError(f"无效的JSON文件路径：{self.path}")
        
        with open(self.path, mode='r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                return content
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON文件解析失败：{e}")
