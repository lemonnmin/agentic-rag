#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   app_.py
@Time    :   2026/03/19 17:23:56
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   界面
"""

import streamlit as st
import sys
import os
import tempfile
import json
import pickle
from pathlib import Path
from agent.controller import OptimizedRAGController
from rag.Embeddings import OpenAIEmbedding
from rag.VectorBase import VectorStore
from tools.rag_search import RagSearchTool  # 用于重新加载向量库后更新工具
from rag.utils import ReadFiles
# ---------- 页面配置 ----------
st.set_page_config(
    page_title="食用菌RAG智能问答系统",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 自定义样式 ----------
st.markdown("""
<style>
    .main-title { font-size: 3rem; font-weight: 700; color: #2e7d32; margin-bottom: 1rem; }
    .step-header { font-size: 1.8rem; font-weight: 600; color: #1b5e20; border-bottom: 2px solid #81c784; margin-top: 1.5rem; margin-bottom: 1rem; }
    .sub-header { font-size: 1.3rem; font-weight: 500; color: #388e3c; margin-top: 1rem; }
    .info-box { background-color: #f1f8e9; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #66bb6a; margin-bottom: 1rem; }
    .score-high { color: #2e7d32; font-weight: 600; }
    .score-medium { color: #f57c00; font-weight: 600; }
    .score-low { color: #d32f2f; font-weight: 600; }
    .metric-card { background-color: #f9f9f9; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ---------- 初始化会话状态 ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None          # VectorStore实例
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "rag_docs" # 默认集合名
if "rag_tool" not in st.session_state:
    st.session_state.rag_tool = None               # RagSearchTool实例，需要重新加载时更新

# ---------- 侧边栏：向量库管理 ----------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/mushroom.png", width=80)
    st.markdown("## 🍄 食用菌RAG系统")
    st.markdown("---")
    
    st.markdown("### 📁 向量库管理")
    
    # 选择操作模式
    mode = st.radio("选择操作", ["加载已有向量库", "上传文件构建新库"])
    
    if mode == "加载已有向量库":
        # 列出storage目录下所有可用的向量库（以.index文件为准）
        storage_dir = os.path.join("rag", "storage")
        if os.path.exists(storage_dir):
            index_files = [f for f in os.listdir(storage_dir) if f.endswith(".index")]
            collections = [f.replace(".index", "") for f in index_files]
            if collections:
                selected = st.selectbox("选择向量库", collections)
                if st.button("加载选中库"):
                    with st.spinner("正在加载向量库..."):
                        try:
                            vs = VectorStore()  # 实例化vectorStore类
                            vs.load_vector(path=storage_dir, collection_name=selected)
                            st.session_state.vector_store = vs # 绑定会话的实例
                            st.session_state.collection_name = selected
                            # 更新RagSearchTool中的向量库
                            if st.session_state.rag_tool is None:
                                st.session_state.rag_tool = RagSearchTool()
                            st.session_state.rag_tool.vector_store = vs
                            st.success(f"成功加载向量库：{selected}")
                        except Exception as e:
                            st.error(f"加载失败：{e}")
            else:
                st.info("storage目录下没有找到向量库文件")
        else:
            st.info("storage目录不存在，请先上传文件构建向量库")
    
    else:  # 上传文件构建新库
        uploaded_files = st.file_uploader("上传文件（支持txt/pdf）", type=["txt", "pdf"], accept_multiple_files=True)
        collection_name = st.text_input("集合名称", value="rag_docs")
        chunk_size = st.slider("文本块大小（字符数）", 200, 2000, 500, step=50)
        chunk_overlap = st.slider("块重叠", 0, 200, 50, step=10)
        
        if st.button("构建向量库") and uploaded_files:
            with st.spinner("正在处理文件并构建向量库..."):
                # 临时保存上传的文件到data目录
                if not os.path.exists("./rag/data"):
                    os.makedirs("./rag/data")
                for file in uploaded_files:
                    file_path = os.path.join("./rag/data", file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                st.success(f"✅ 上传{len(uploaded_files)}个文件到data目录！")
                # 加载并构建向量库
                docs = ReadFiles('./rag/data').get_content(max_token_len=600, cover_content=150)
                
                # 初始化向量库
                vs = VectorStore(docs)
                embedding_model = OpenAIEmbedding(is_api=True)  # 默认使用API，您可调整为本地
                vectors = vs.get_vector(embedding_model)
                if vectors:
                    vs.persist(path='./rag/storage')  # 保存到storage目录
                    st.session_state.vector_store = vs
                    st.session_state.collection_name = collection_name
                    if st.session_state.rag_tool is None:
                        st.session_state.rag_tool = RagSearchTool()
                    st.session_state.rag_tool.vector_store = vs
                    st.success(f"向量库构建成功！共 {len(docs)} 个文本块")
                else:
                    st.error("向量生成失败")
    
    st.markdown("---")
    st.markdown("### ⚙️ 系统参数")
    max_retries = st.slider("最大优化重试次数", 0, 3, 2, 1)
    show_full = st.checkbox("显示完整融合结果", value=False)
    

# ---------- 主界面 ----------
st.markdown('<p class="main-title">🍄 食用菌种植智能问答系统</p>', unsafe_allow_html=True)
st.markdown("输入您的问题，系统将通过多代理工作流为您生成专业回答，并展示每个环节的详细结果。")

# 检查向量库是否已加载
if st.session_state.vector_store is None:
    st.warning("请先在侧边栏加载或构建向量库")
else:
    # 显示当前向量库信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("向量库名称", st.session_state.collection_name)
    with col2:
        stats = st.session_state.vector_store.get_collection_stats()
        st.metric("文档块数量", stats["文档数量"])
    with col3:
        st.metric("向量维度", stats["向量维度"])

    # 查询输入
    query = st.text_input("**您的问题**", placeholder="例如：北京种植香菇需要控制哪些温度和湿度条件？")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        run_button = st.button("🚀 开始分析", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ 清除", use_container_width=True)

    if clear_button:
        st.rerun()

    if run_button and query:
        # 初始化控制器
        controller = OptimizedRAGController()
        
        # 执行流程（带进度提示）
        with st.spinner("正在执行全流程，请稍候..."):
            try:
                result = controller.run(query, max_retries=max_retries)
            except Exception as e:
                st.error(f"执行过程中发生错误：{str(e)}")
                st.stop()
        
        # 检查是否成功
        if not result.get("success"):
            st.error(f"处理失败：{result.get('error', '未知错误')}")
            st.stop()
        
        # 成功，开始展示结果
        st.success("✅ 处理完成！")
        
        # 总体信息
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最终评分", f"{result.get('final_score', 0)}/5")
        with col2:
            st.metric("优化重试次数", result.get('retry_count', 0))
        with col3:
            tools = result.get("reasoning", {}).get("called_tools", [])
            st.metric("调用工具", ", ".join(tools) if tools else "无")
        
        # ========== 1. 意图理解环节 ==========
        st.markdown('<p class="step-header">🔍 第一步：意图理解</p>', unsafe_allow_html=True)
        intent = result.get("intent", {})
        if intent:
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**意图类型**：`{intent.get('intent_type')}`")
                    st.markdown(f"**领域**：`{intent.get('domain')}`")
                    st.markdown(f"**复杂度**：`{intent.get('complexity')}`")
                with col2:
                    st.markdown(f"**城市**：`{intent.get('city', '无')}`")
                    st.markdown(f"**关键词**：`{', '.join(intent.get('keywords', []))}`")
                if intent.get('sub_tasks'):
                    st.markdown("**子任务**：")
                    for task in intent['sub_tasks']:
                        st.markdown(f"- {task}")
        else:
            st.info("无意图数据")
        
        # ========== 2. 检索规划环节 ==========
        st.markdown('<p class="step-header">📋 第二步：检索规划</p>', unsafe_allow_html=True)
        planner = result.get("planner", {})
        if planner:
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**检索模式**：`{planner.get('mode')}`")
                    st.markdown(f"**检索器**：`{', '.join(planner.get('retrievers', []))}`")
                with col2:
                    st.markdown(f"**top_k**：`{planner.get('top_k')}`")
                    st.markdown(f"**多轮检索**：`{planner.get('multi_round')}`")
                with col3:
                    st.markdown(f"**重排序**：`{planner.get('rerank', False)}`")
                if planner.get('expand_keywords'):
                    st.markdown("**扩展关键词**：")
                    st.markdown(f"`{', '.join(planner['expand_keywords'])}`")
        else:
            st.info("无规划数据")
        
        # ========== 3. 推理代理结果 ==========
        st.markdown('<p class="step-header">🤔 第三步：推理代理（RAG + 工具调用）</p>', unsafe_allow_html=True)
        reasoning = result.get("reasoning", {})
        if reasoning:
            # 原始答案与优化答案
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始答案**")
                st.markdown(f'<div class="info-box">{reasoning.get("raw_answer", "无")}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown("**优化后答案**")
                st.markdown(f'<div class="info-box">{reasoning.get("optimized_answer", "无")}</div>', unsafe_allow_html=True)
            
            # 检索轮次与工具调用
            st.markdown(f"**检索轮数**：`{reasoning.get('retrieve_rounds')}`  |  **调用工具**：`{reasoning.get('called_tools')}`")
            
            # 融合结果展示（可展开）
            fused = reasoning.get("fused_results", [])
            if fused:
                with st.expander(f"查看融合结果（共 {len(fused)} 条）"):
                    if show_full:
                        for i, r in enumerate(fused):
                            st.markdown(f"**{i+1}. 来源：{r.get('source')}**")
                            st.markdown(r.get('content', ''))
                            st.markdown("---")
                    else:
                        # 只显示前3条摘要
                        for i, r in enumerate(fused[:3]):
                            st.markdown(f"**{i+1}. 来源：{r.get('source')}**")
                            content = r.get('content', '')
                            st.markdown(content[:300] + "..." if len(content) > 300 else content)
                            st.markdown("---")
                        if len(fused) > 3:
                            st.info(f"还有 {len(fused)-3} 条结果未显示，请勾选侧边栏『显示完整融合结果』查看全部。")
        else:
            st.info("无推理结果")
        
        # ========== 4. 评估结果 ==========
        st.markdown('<p class="step-header">📊 第四步：系统评估</p>', unsafe_allow_html=True)
        evaluation = result.get("evaluation", {})
        if evaluation:
            # 分数卡片布局
            score_items = [
                ("检索相关性", evaluation.get('retrieval_relevance', 0)),
                ("答案准确性", evaluation.get('answer_accuracy', 0)),
                ("答案完整性", evaluation.get('answer_completeness', 0)),
                ("推理有效性", evaluation.get('reasoning_effectiveness', 0)),
                ("工具调用合理性", evaluation.get('tool_call_appropriateness', 0)),
                ("结果融合质量", evaluation.get('result_fusion_quality', 0)),
                ("答案优化效果", evaluation.get('answer_optimization_effect', 0))
            ]
            # 分成两行显示
            row1 = score_items[:4]
            row2 = score_items[4:]
            
            for row in [row1, row2]:
                cols = st.columns(len(row))
                for col, (label, score) in zip(cols, row):
                    with col:
                        if score >= 4:
                            color = "score-high"
                        elif score >= 2:
                            color = "score-medium"
                        else:
                            color = "score-low"
                        st.markdown(f'<div class="metric-card"><span style="font-size:1rem;">{label}</span><br><span class="{color}" style="font-size:2rem;">{score}</span><span style="font-size:1rem;">/5</span></div>', unsafe_allow_html=True)
            
            # 优化建议
            suggestion = evaluation.get('suggestion', '')
            if suggestion:
                st.markdown("**💡 优化建议**")
                st.markdown(f'<div class="info-box">{suggestion}</div>', unsafe_allow_html=True)
        else:
            st.info("无评估数据")
        
        # ========== 5. 优化建议列表（如果存在） ==========
        suggestions = result.get("optimization_suggestions", [])
        if suggestions:
            st.markdown('<p class="step-header">⚙️ 第五步：自动优化建议</p>', unsafe_allow_html=True)
            for i, s in enumerate(suggestions):
                st.markdown(f"**{i+1}.** {s}")
        
        # 优化报告（如果有重试）
        if result.get("optimization_report"):
            with st.expander("查看优化报告"):
                st.markdown(result["optimization_report"])
        
        # 显示完整JSON（调试用）
        with st.expander("查看原始返回数据（JSON）"):
            st.json(result)