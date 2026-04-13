#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   rag_ui.py
@Time    :   2026/03/10 23:13:33
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   
"""

import streamlit as st
import os
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# ---------------------- 页面基础配置 ----------------------
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="📚",
    layout="wide"  # 宽屏布局
)

# 侧边栏：配置区
with st.sidebar:
    st.title("⚙️ 配置中心")
    
    # 1. OpenAI API Key配置（必选）
    api_key = st.text_input("OpenAI API Key", type="password", 
                           placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if api_key:
        # 设置环境变量，供Embedding和LLM调用
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 2. 文档加载方式选择
    load_mode = st.radio("文档加载方式", ["加载本地data目录", "上传文件"])
    
    # 3. 向量库配置
    st.divider()
    st.subheader("向量库管理")
    build_vector = st.button("📝 重新构建向量库", type="primary")
    load_local_vector = st.button("📂 加载本地向量库")

# ---------------------- 主页面：问答区 ----------------------
st.title("📚 RAG智能问答系统")
st.divider()

# 初始化会话状态（保存向量库、Embedding、LLM实例，避免重复初始化）
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

# ---------------------- 第一步：初始化模型和向量库 ----------------------
def init_models():
    """初始化Embedding和LLM模型"""
    if not api_key:
        st.error("请先在侧边栏输入OpenAI API Key！")
        return False
    try:
        # 初始化Embedding模型
        st.session_state.embedding_model = OpenAIEmbedding()
        # 初始化LLM模型
        st.session_state.chat_model = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
        st.success("✅ 模型初始化成功！")
        return True
    except Exception as e:
        st.error(f"模型初始化失败：{str(e)}")
        return False

def init_vector_store(docs):
    """初始化向量库并生成向量"""
    if not st.session_state.embedding_model:
        st.error("请先初始化模型！")
        return
    try:
        # 创建向量库实例
        vector = VectorStore(docs)
        # 生成向量
        with st.spinner("🔄 正在计算文档嵌入向量..."):
            vector.get_vector(EmbeddingModel=st.session_state.embedding_model)
        # 保存到本地
        vector.persist(path='storage')
        st.session_state.vector_store = vector
        st.success("✅ 向量库构建并保存成功！")
    except Exception as e:
        st.error(f"向量库构建失败：{str(e)}")

# ---------------------- 处理侧边栏按钮事件 ----------------------
# 1. 重新构建向量库
if build_vector:
    if not init_models():
        pass
    else:
        # 根据加载方式获取文档
        if load_mode == "加载本地data目录":
            if not os.path.exists("./data"):
                st.error("❌ 本地data目录不存在！")
            else:
                with st.spinner("📂 正在加载data目录文档..."):
                    docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
                init_vector_store(docs)
        else:  # 上传文件
            st.warning("请先在下方上传文件，再点击「重新构建向量库」！")

# 2. 加载本地向量库
if load_local_vector:
    if not init_models():
        pass
    else:
        try:
            # 核心修改：指定storage_path，与构建时保持一致
            vector = VectorStore(storage_path='./storage')
            with st.spinner("📂 正在加载本地向量库..."):
                vector.load_vector('./storage')
            st.session_state.vector_store = vector
            st.success("✅ 本地向量库加载成功！")  # 这里的emoji不影响（Streamlit前端渲染）
        except Exception as e:
            st.error(f"加载本地向量库失败：{str(e)}（请先构建并保存向量库）")

# ---------------------- 第二步：文件上传（可选） ----------------------
if load_mode == "上传文件":
    st.subheader("📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持txt、md、docx等格式（需确保ReadFiles能解析）",
        accept_multiple_files=True
    )
    if uploaded_files and build_vector:
        # 临时保存上传的文件到data目录
        if not os.path.exists("./data"):
            os.makedirs("./data")
        for file in uploaded_files:
            file_path = os.path.join("./data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ 上传{len(uploaded_files)}个文件到data目录！")
        # 加载并构建向量库
        docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
        init_vector_store(docs)

# ---------------------- 第三步：问答交互 ----------------------
st.divider()
st.subheader("❓ 智能问答")
question = st.text_input("请输入你的问题", placeholder="例如：RAG的原理是什么？")
submit_btn = st.button("🚀 提交问题", type="primary")

if submit_btn and question:
    # 校验前置条件
    if not st.session_state.vector_store:
        st.error("❌ 请先构建/加载向量库！")
    elif not st.session_state.chat_model:
        st.error("❌ 请先初始化模型！")
    else:
        try:
            # 1. 检索相似文档（k建议设为3-5，展示更多结果）
            with st.spinner("🔍 正在检索相关文档..."):
                # 注意：这里k值改为3（可根据需求调整），保留所有检索结果列表
                content_list = st.session_state.vector_store.query(
                    question, 
                    EmbeddingModel=st.session_state.embedding_model, 
                    k=5  # 改为大于1的值，才能返回多个结果
                )
                
                # 处理无检索结果的情况
                if not content_list:
                    st.warning("⚠️ 未检索到相关文档！")
                    content_combined = ""  # 传给LLM的空内容
                else:
                    # 2. 展示所有检索结果（带序号，折叠面板）
                    with st.expander("📄 检索到的参考文档（共{}条）".format(len(content_list)), expanded=False):
                        # 遍历所有结果，逐条展示
                        for idx, doc in enumerate(content_list, 1):
                            st.markdown(f"#### 📑 参考文档 {idx}")
                            st.write(doc)
                            # 每条文档之间加分隔线，更清晰
                            if idx < len(content_list):
                                st.divider()
                    
                    # 3. 合并所有检索结果（传给LLM生成回答）
                    # 用分隔符区分不同文档，避免LLM混淆
                    content_combined = "\n\n--- 参考文档分隔线 ---\n\n".join(content_list)
            
            # 2. 调用LLM生成回答（传入合并后的所有文档）
            with st.spinner("🤖 AI正在思考回答..."):
                # 注意：这里传入的是合并后的content_combined，而非单个文档
                answer = st.session_state.chat_model.chat(question, [], content_combined)
            
            # 3. 展示回答
            st.success("✅ 回答生成完成！")
            st.markdown("### 🎯 回答结果")
            st.write(answer)
        except Exception as e:
            st.error(f"问答失败：{str(e)}")