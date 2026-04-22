#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行 50 条测试问题并导出实验结果。

默认支持四种实验方法：
- LLM_ONLY
- BASIC_RAG
- RAG_RERANK
- AGENTIC_RAG

导出内容：
- 明细 CSV
- 方法汇总 CSV
- 分类汇总 CSV
- 完整 JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")


SUPPORTED_METHODS = ("LLM_ONLY", "BASIC_RAG", "RAG_RERANK", "AGENTIC_RAG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行 Agentic RAG 实验并导出结果")
    parser.add_argument(
        "--questions",
        default=str(REPO_ROOT / "experiments" / "test_questions_50.json"),
        help="测试问题集 JSON 路径",
    )
    parser.add_argument(
        "--methods",
        default="LLM_ONLY,BASIC_RAG,RAG_RERANK,AGENTIC_RAG",
        help="实验方法，逗号分隔",
    )
    parser.add_argument(
        "--collection-name",
        default="rag_docs",
        help="向量库集合名",
    )
    parser.add_argument(
        "--storage-dir",
        default=str(REPO_ROOT / "rag" / "storage"),
        help="FAISS 存储目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "results"),
        help="实验结果输出目录",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Agentic RAG 最大优化重试次数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅运行前 N 条问题，0 表示全部运行",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="输出文件名前缀，默认使用时间戳",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-32B-Instruct",
        help="LLM_ONLY 基线使用的模型名",
    )
    return parser.parse_args()


def load_questions(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError(f"测试集格式错误，期望为列表：{path}")
    if limit > 0:
        return questions[:limit]
    return questions


def ensure_vector_store(collection_name: str, storage_dir: Path):
    from rag.VectorBase import VectorStore
    from tools.rag_search import RagSearchTool

    vector_store = VectorStore()
    vector_store.load_vector(path=str(storage_dir), collection_name=collection_name)
    RagSearchTool.set_shared_vector_db(vector_store)
    return vector_store


def create_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def compute_final_score(evaluation_data: Dict[str, Any]) -> float:
    if not evaluation_data:
        return 0.0

    evaluation_mode = evaluation_data.get("evaluation_mode", "standard")
    if evaluation_mode == "boundary":
        score_keys = [
            "boundary_recognition",
            "scope_compliance",
            "response_clarity",
            "helpful_redirection",
            "tool_call_appropriateness",
        ]
    else:
        score_keys = [
            "retrieval_relevance",
            "answer_accuracy",
            "answer_completeness",
        ]

    score = sum(float(evaluation_data.get(key, 0) or 0) for key in score_keys) / len(score_keys)
    return round(score, 2)


def normalize_tool_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def expected_tool_matched(expected_tool: str, called_tools: List[str]) -> bool:
    expected = (expected_tool or "").strip()
    if not expected:
        return len(called_tools) == 0
    return expected in called_tools


def make_baseline_plan(method: str) -> Dict[str, Any]:
    plan = {
        "success": True,
        "data": {
            "retrievers": ["vector"],
            "top_k": 3,
            "rerank": False,
            "multi_round": False,
            "expand_keywords": [],
            "mode": "baseline",
            "sub_tasks": [],
        },
        "error": None,
    }
    if method == "RAG_RERANK":
        plan["data"]["retrievers"] = ["vector", "bm25"]
        plan["data"]["top_k"] = 5
        plan["data"]["rerank"] = True
        plan["data"]["mode"] = "baseline_rerank"
    elif method == "BASIC_RAG":
        plan["data"]["mode"] = "baseline_basic"
    else:
        raise ValueError(f"不支持的基线方法：{method}")
    return plan


def run_llm_only(question: str, model: str) -> Dict[str, Any]:
    client = create_openai_client()
    prompt = (
        "你是一名食用菌种植问答助手，请仅基于自身已有知识回答用户问题。"
        "要求使用中文，回答尽量专业、简洁、有条理；"
        "如果无法确定，请明确说明不确定，不要编造。\n\n"
        f"用户问题：{question}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )
    answer = response.choices[0].message.content or ""
    return {
        "success": True,
        "fused_results": [],
        "raw_answer": answer,
        "optimized_answer": answer,
        "retrieve_rounds": 0,
        "called_tools": [],
        "tool_results": [],
        "error": None,
    }


def build_context_text(docs: List[str]) -> str:
    return "\n".join(doc for doc in docs if isinstance(doc, str) and doc.strip())


def build_fused_results(docs: List[str], source: str) -> List[Dict[str, Any]]:
    fused_results: List[Dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        fused_results.append(
            {
                "content": doc,
                "source": source,
                "priority": 0,
                "rank": index,
            }
        )
    return fused_results


def run_basic_rag(question: str) -> Dict[str, Any]:
    from tools.rag_search import RagSearchTool

    rag_tool = RagSearchTool(mode="simple")
    docs = rag_tool.retrieve(question)
    context_text = build_context_text(docs)
    answer = rag_tool.llm.chat(
        prompt=question,
        history=[],
        content=context_text,
    )

    return {
        "success": True,
        "fused_results": build_fused_results(docs, "rag_db"),
        "raw_answer": answer,
        "optimized_answer": answer,
        "retrieve_rounds": 1 if docs else 0,
        "called_tools": [],
        "tool_results": [],
        "error": None,
        "baseline_mode": "simple",
    }


def run_fixed_rag(question: str) -> Dict[str, Any]:
    from rag.Embeddings import OpenAIEmbedding
    from rag.LLM import OpenAIChat
    from tools.rag_search import RagSearchTool

    rag_tool = RagSearchTool(mode="simple")
    vector_db = rag_tool.vector_db
    embedding = OpenAIEmbedding()
    llm = OpenAIChat()

    docs = vector_db.query(
        query=question,
        EmbeddingModel=embedding,
        k=5,
        retrievers=["vector", "bm25"],
        rerank=True,
        expand_keywords=None,
        history_queries=None,
    )
    context_text = build_context_text(docs)
    answer = llm.chat(
        prompt=question,
        history=[],
        content=context_text,
    )

    return {
        "success": True,
        "fused_results": build_fused_results(docs, "rag_fixed"),
        "raw_answer": answer,
        "optimized_answer": answer,
        "retrieve_rounds": 1 if docs else 0,
        "called_tools": [],
        "tool_results": [],
        "error": None,
        "baseline_mode": "fixed_rerank",
    }


def run_agentic(
    question: str,
    controller,
    max_retries: int,
    collection_name: str,
    evaluation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return controller.run(
        question,
        max_retries=max_retries,
        collection_name=collection_name,
        evaluation_context=evaluation_context,
    )


def evaluate_result(
    question: str,
    reasoning_result: Dict[str, Any],
    evaluator,
    evaluation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evaluation = evaluator.evaluate(question, reasoning_result, evaluation_context or {})
    return evaluation.get("data", {}) if evaluation else {}


def build_record(
    run_id: int,
    method: str,
    question_item: Dict[str, Any],
    result: Dict[str, Any],
    response_time_s: float,
) -> Dict[str, Any]:
    question_id = question_item.get("id", "")
    category = question_item.get("category", "")
    question = question_item.get("question", "")
    expected_tool = question_item.get("expected_tool", "")

    if method == "AGENTIC_RAG":
        evaluation_data = result.get("evaluation", {}) or {}
        reasoning = result.get("reasoning", {}) or {}
        called_tools = normalize_tool_list(reasoning.get("called_tools", []))
        retrieve_rounds = reasoning.get("retrieve_rounds", 0)
        retry_count = result.get("retry_count", 0)
        final_score = result.get("final_score", compute_final_score(evaluation_data))
        success = bool(result.get("success", False))
        notes = result.get("error") or ""
    else:
        evaluation_data = result.get("evaluation", {}) or {}
        called_tools = normalize_tool_list(result.get("called_tools", []))
        retrieve_rounds = result.get("retrieve_rounds", 0)
        retry_count = 0
        final_score = compute_final_score(evaluation_data)
        success = bool(result.get("success", False))
        notes = result.get("error") or ""

    if not notes and not expected_tool_matched(expected_tool, called_tools):
        notes = f"工具调用与预期不一致，expected={expected_tool or 'NONE'}"

    record = {
        "run_id": run_id,
        "method": method,
        "question_id": question_id,
        "category": category,
        "question": question,
        "expected_tool": expected_tool,
        "called_tools": "|".join(called_tools),
        "retrieve_rounds": retrieve_rounds,
        "retry_count": retry_count,
        "final_score": round(float(final_score), 2) if final_score is not None else 0.0,
        "evaluation_mode": evaluation_data.get("evaluation_mode", "standard"),
        "retrieval_relevance": evaluation_data.get("retrieval_relevance", 0),
        "answer_accuracy": evaluation_data.get("answer_accuracy", 0),
        "answer_completeness": evaluation_data.get("answer_completeness", 0),
        "reasoning_effectiveness": evaluation_data.get("reasoning_effectiveness", 0),
        "tool_call_appropriateness": evaluation_data.get("tool_call_appropriateness", 0),
        "result_fusion_quality": evaluation_data.get("result_fusion_quality", 0),
        "answer_optimization_effect": evaluation_data.get("answer_optimization_effect", 0),
        "boundary_recognition": evaluation_data.get("boundary_recognition", 0),
        "scope_compliance": evaluation_data.get("scope_compliance", 0),
        "response_clarity": evaluation_data.get("response_clarity", 0),
        "helpful_redirection": evaluation_data.get("helpful_redirection", 0),
        "response_time_s": round(response_time_s, 3),
        "success": success,
        "notes": notes,
    }
    return record


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(statistics.mean(values), 3)


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    methods = sorted({record["method"] for record in records})
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        expected_records = [record for record in method_records if record["expected_tool"] or record["called_tools"]]
        tool_matches = [
            1 if expected_tool_matched(record["expected_tool"], normalize_tool_list(record["called_tools"].split("|") if record["called_tools"] else [])) else 0
            for record in method_records
        ]
        summaries.append(
            {
                "method": method,
                "samples": len(method_records),
                "success_rate": mean_or_zero(1 if record["success"] else 0 for record in method_records),
                "avg_final_score": mean_or_zero(record["final_score"] for record in method_records),
                "avg_retrieval_relevance": mean_or_zero(record["retrieval_relevance"] for record in method_records),
                "avg_answer_accuracy": mean_or_zero(record["answer_accuracy"] for record in method_records),
                "avg_answer_completeness": mean_or_zero(record["answer_completeness"] for record in method_records),
                "avg_reasoning_effectiveness": mean_or_zero(record["reasoning_effectiveness"] for record in method_records),
                "avg_tool_call_appropriateness": mean_or_zero(record["tool_call_appropriateness"] for record in method_records),
                "avg_result_fusion_quality": mean_or_zero(record["result_fusion_quality"] for record in method_records),
                "avg_answer_optimization_effect": mean_or_zero(record["answer_optimization_effect"] for record in method_records),
                "avg_response_time_s": mean_or_zero(record["response_time_s"] for record in method_records),
                "avg_retrieve_rounds": mean_or_zero(record["retrieve_rounds"] for record in method_records),
                "avg_retry_count": mean_or_zero(record["retry_count"] for record in method_records),
                "tool_expectation_match_rate": mean_or_zero(tool_matches),
                "optimization_trigger_rate": mean_or_zero(1 if record["retry_count"] > 0 else 0 for record in method_records),
                "tool_expectation_samples": len(expected_records),
            }
        )
    return summaries


def summarize_by_category(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    groups = sorted({(record["method"], record["category"]) for record in records})
    for method, category in groups:
        subset = [record for record in records if record["method"] == method and record["category"] == category]
        summary_rows.append(
            {
                "method": method,
                "category": category,
                "samples": len(subset),
                "avg_final_score": mean_or_zero(record["final_score"] for record in subset),
                "avg_answer_accuracy": mean_or_zero(record["answer_accuracy"] for record in subset),
                "avg_answer_completeness": mean_or_zero(record["answer_completeness"] for record in subset),
                "avg_reasoning_effectiveness": mean_or_zero(record["reasoning_effectiveness"] for record in subset),
                "avg_response_time_s": mean_or_zero(record["response_time_s"] for record in subset),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_single_method(
    method: str,
    question_item: Dict[str, Any],
    controller,
    evaluator,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = question_item["question"]
    start = time.perf_counter()

    if method == "LLM_ONLY":
        base_result = run_llm_only(question, args.llm_model)
        evaluation_data = evaluate_result(question, base_result, evaluator, question_item)
        result = deepcopy(base_result)
        result["evaluation"] = evaluation_data
    elif method == "BASIC_RAG":
        base_result = run_basic_rag(question)
        evaluation_data = evaluate_result(question, base_result, evaluator, question_item)
        result = deepcopy(base_result)
        result["evaluation"] = evaluation_data
    elif method == "RAG_RERANK":
        base_result = run_fixed_rag(question)
        evaluation_data = evaluate_result(question, base_result, evaluator, question_item)
        result = deepcopy(base_result)
        result["evaluation"] = evaluation_data
    elif method == "AGENTIC_RAG":
        if controller is None:
            raise ValueError("AGENTIC_RAG 模式缺少 controller")
        result = run_agentic(question, controller, args.max_retries, args.collection_name, question_item)
    else:
        raise ValueError(f"未知方法：{method}")

    elapsed = time.perf_counter() - start
    result["response_time_s"] = elapsed
    return result


def main() -> None:
    args = parse_args()

    methods = [item.strip().upper() for item in args.methods.split(",") if item.strip()]
    invalid = [method for method in methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"存在不支持的方法：{invalid}，可选值：{SUPPORTED_METHODS}")

    questions_path = Path(args.questions)
    storage_dir = Path(args.storage_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path, args.limit)

    if any(method != "LLM_ONLY" for method in methods):
        print(f"[INFO] 加载向量库：collection={args.collection_name}")
        ensure_vector_store(args.collection_name, storage_dir)

    from agent.controller import OptimizedRAGController
    from agent.evaluation import EvaluationGraphAgent

    controller = OptimizedRAGController() if "AGENTIC_RAG" in methods else None
    evaluator = EvaluationGraphAgent()

    run_label = args.run_label.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    records: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []

    total_runs = len(methods) * len(questions)
    current = 0

    for method in methods:
        for question_item in questions:
            current += 1
            question_id = question_item.get("id", "")
            print(f"[{current}/{total_runs}] {method} -> {question_id}")

            try:
                result = run_single_method(method, question_item, controller, evaluator, args)
                record = build_record(
                    run_id=current,
                    method=method,
                    question_item=question_item,
                    result=result,
                    response_time_s=result.get("response_time_s", 0.0),
                )
            except Exception as exc:
                record = {
                    "run_id": current,
                    "method": method,
                    "question_id": question_item.get("id", ""),
                    "category": question_item.get("category", ""),
                    "question": question_item.get("question", ""),
                    "expected_tool": question_item.get("expected_tool", ""),
                    "called_tools": "",
                    "retrieve_rounds": 0,
                    "retry_count": 0,
                    "final_score": 0.0,
                    "retrieval_relevance": 0,
                    "answer_accuracy": 0,
                    "answer_completeness": 0,
                    "reasoning_effectiveness": 0,
                    "tool_call_appropriateness": 0,
                    "result_fusion_quality": 0,
                    "answer_optimization_effect": 0,
                    "response_time_s": 0.0,
                    "success": False,
                    "notes": f"执行异常：{exc}",
                }
                result = {"success": False, "error": str(exc)}

            records.append(record)
            full_results.append(
                {
                    "record": record,
                    "question_item": question_item,
                    "result": result,
                }
            )

    summary_rows = summarize_records(records)
    category_summary_rows = summarize_by_category(records)

    detail_csv = output_dir / f"{run_label}_detailed.csv"
    summary_csv = output_dir / f"{run_label}_summary.csv"
    category_summary_csv = output_dir / f"{run_label}_category_summary.csv"
    full_json = output_dir / f"{run_label}_full.json"

    write_csv(detail_csv, records)
    write_csv(summary_csv, summary_rows)
    write_csv(category_summary_csv, category_summary_rows)
    full_json.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "collection_name": args.collection_name,
                "methods": methods,
                "question_count": len(questions),
                "records": full_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[OK] 实验完成")
    print(f"- 明细结果: {detail_csv}")
    print(f"- 方法汇总: {summary_csv}")
    print(f"- 分类汇总: {category_summary_csv}")
    print(f"- 完整 JSON: {full_json}")


if __name__ == "__main__":
    main()
