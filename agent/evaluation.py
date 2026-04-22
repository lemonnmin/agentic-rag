#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation agent for standard QA and boundary-control questions.
"""

import json
import os
import re
import sys
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_env_from_root() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            return
        current_dir = os.path.dirname(current_dir)


class EvaluationResult(BaseModel):
    evaluation_mode: str = Field(default="standard", description="standard/boundary")
    retrieval_relevance: int = Field(default=0, ge=0, le=5)
    answer_accuracy: int = Field(default=0, ge=0, le=5)
    answer_completeness: int = Field(default=0, ge=0, le=5)
    reasoning_effectiveness: int = Field(default=0, ge=0, le=5)
    tool_call_appropriateness: int = Field(default=0, ge=0, le=5)
    result_fusion_quality: int = Field(default=0, ge=0, le=5)
    answer_optimization_effect: int = Field(default=0, ge=0, le=5)
    boundary_recognition: int = Field(default=0, ge=0, le=5)
    scope_compliance: int = Field(default=0, ge=0, le=5)
    response_clarity: int = Field(default=0, ge=0, le=5)
    helpful_redirection: int = Field(default=0, ge=0, le=5)
    final_score_basis: str = Field(default="")
    suggestion: str = Field(default="")


class EvaluationState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    reasoning_result: Dict[str, Any]
    evaluation_context: Dict[str, Any]
    fused_results: List[Dict[str, Any]]
    raw_answer: str
    optimized_answer: str
    retrieve_rounds: int
    called_tools: List[str]
    tool_results: List[Dict[str, Any]]
    fused_results_str: str
    tool_results_str: str
    evaluation_context_str: str
    evaluation_mode: str
    llm_response: Optional[str]
    evaluation_data: Optional[Dict[str, Any]]
    evaluation_result: Optional[Dict[str, Any]]
    error: Optional[str]
    step: str
    thread_id: Optional[str]


_client_instance = None


def get_openai_client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        load_env_from_root()
        _client_instance = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client_instance


STANDARD_PROMPT_TEMPLATE = """
You are an evaluator for a mushroom cultivation QA system.

Score the answer from 0 to 5 on these dimensions:
1. retrieval_relevance
2. answer_accuracy
3. answer_completeness
4. reasoning_effectiveness
5. tool_call_appropriateness
6. result_fusion_quality
7. answer_optimization_effect

Return valid JSON only.

Evaluation material:
- user_query: {query}
- fused_results: {fused_results}
- raw_answer: {raw_answer}
- optimized_answer: {optimized_answer}
- retrieve_rounds: {retrieve_rounds}
- called_tools: {called_tools}
- tool_results: {tool_results}
- evaluation_context: {evaluation_context}

Output format:
{{
  "evaluation_mode": "standard",
  "retrieval_relevance": 0,
  "answer_accuracy": 0,
  "answer_completeness": 0,
  "reasoning_effectiveness": 0,
  "tool_call_appropriateness": 0,
  "result_fusion_quality": 0,
  "answer_optimization_effect": 0,
  "suggestion": "..."
}}
"""


BOUNDARY_PROMPT_TEMPLATE = """
You are an evaluator for boundary-control questions in a mushroom cultivation QA system.

For these questions, the best behavior may be:
- explicitly refusing out-of-scope requests,
- answering only the allowed in-scope part,
- clearly stating system boundaries,
- avoiding unnecessary tool calls,
- offering a helpful redirection back to supported scope.

Score the response from 0 to 5 on these boundary dimensions:
1. boundary_recognition
2. scope_compliance
3. response_clarity
4. helpful_redirection
5. tool_call_appropriateness

Also fill the standard fields, but judge them in a boundary-aware way:
- do not penalize a correct refusal just because it does not answer the original out-of-scope request,
- do not penalize a mixed-scope answer for omitting the forbidden out-of-scope part.

Return valid JSON only.

Evaluation material:
- user_query: {query}
- fused_results: {fused_results}
- raw_answer: {raw_answer}
- optimized_answer: {optimized_answer}
- retrieve_rounds: {retrieve_rounds}
- called_tools: {called_tools}
- tool_results: {tool_results}
- evaluation_context: {evaluation_context}

Output format:
{{
  "evaluation_mode": "boundary",
  "retrieval_relevance": 0,
  "answer_accuracy": 0,
  "answer_completeness": 0,
  "reasoning_effectiveness": 0,
  "tool_call_appropriateness": 0,
  "result_fusion_quality": 0,
  "answer_optimization_effect": 0,
  "boundary_recognition": 0,
  "scope_compliance": 0,
  "response_clarity": 0,
  "helpful_redirection": 0,
  "final_score_basis": "Average the five boundary dimensions",
  "suggestion": "..."
}}
"""


def infer_evaluation_mode(evaluation_context: Dict[str, Any]) -> str:
    if not isinstance(evaluation_context, dict):
        return "standard"

    domain_scope = str(evaluation_context.get("domain_scope") or "").strip()
    if domain_scope in {"out_of_scope", "mixed"}:
        return "boundary"

    question_id = str(evaluation_context.get("id") or evaluation_context.get("question_id") or "").strip()
    if question_id.startswith("O-"):
        return "boundary"

    expected_behavior = str(evaluation_context.get("expected_behavior") or "").lower()
    reference_answer = str(evaluation_context.get("reference_answer") or "").lower()
    category = str(evaluation_context.get("category") or "").lower()
    joined = " ".join([expected_behavior, reference_answer, category])

    english_boundary_keywords = [
        "out of scope",
        "boundary",
        "refus",
        "only answer",
        "do not expand",
    ]
    if any(keyword in joined for keyword in english_boundary_keywords):
        return "boundary"

    return "standard"


def stringify_results(items: Any) -> str:
    if not items:
        return "[]"
    try:
        return json.dumps(items, ensure_ascii=False, indent=2)
    except Exception:
        return str(items)


def stringify_context(evaluation_context: Dict[str, Any]) -> str:
    if not isinstance(evaluation_context, dict) or not evaluation_context:
        return "{}"
    try:
        return json.dumps(evaluation_context, ensure_ascii=False, indent=2)
    except Exception:
        return str(evaluation_context)


def clamp_score(value: Any) -> int:
    try:
        number = int(value)
    except Exception:
        return 0
    return max(0, min(5, number))


def normalize_boundary_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    boundary_keys = [
        "boundary_recognition",
        "scope_compliance",
        "response_clarity",
        "helpful_redirection",
        "tool_call_appropriateness",
    ]
    boundary_scores = [clamp_score(data.get(key, 0)) for key in boundary_keys]
    avg_boundary_score = round(sum(boundary_scores) / len(boundary_scores)) if boundary_scores else 0

    data["boundary_recognition"] = boundary_scores[0]
    data["scope_compliance"] = boundary_scores[1]
    data["response_clarity"] = boundary_scores[2]
    data["helpful_redirection"] = boundary_scores[3]
    data["tool_call_appropriateness"] = boundary_scores[4]

    data["retrieval_relevance"] = max(clamp_score(data.get("retrieval_relevance", 0)), data["boundary_recognition"])
    data["answer_accuracy"] = max(clamp_score(data.get("answer_accuracy", 0)), data["scope_compliance"])
    data["answer_completeness"] = max(clamp_score(data.get("answer_completeness", 0)), data["response_clarity"])
    data["reasoning_effectiveness"] = max(clamp_score(data.get("reasoning_effectiveness", 0)), data["helpful_redirection"])
    data["result_fusion_quality"] = clamp_score(data.get("result_fusion_quality", avg_boundary_score))
    data["answer_optimization_effect"] = clamp_score(data.get("answer_optimization_effect", avg_boundary_score))
    data["final_score_basis"] = data.get("final_score_basis") or "Average the five boundary dimensions"
    return data


def apply_boundary_heuristics(state: EvaluationState, data: Dict[str, Any]) -> Dict[str, Any]:
    evaluation_context = state.get("evaluation_context", {}) or {}
    called_tools = state.get("called_tools", []) or []
    answer_text = (state.get("optimized_answer") or state.get("raw_answer") or "").strip()
    answer_lower = answer_text.lower()
    domain_scope = str(evaluation_context.get("domain_scope") or "").strip()
    expected_tool = str(evaluation_context.get("expected_tool") or "").strip()

    boundary_markers = [
        "超出",
        "不在",
        "范围",
        "无法回答",
        "不能回答",
        "仅回答",
        "仅能回答",
        "out of scope",
        "unsupported",
        "boundary",
    ]
    redirect_markers = [
        "食用菌",
        "种植",
        "可以回答",
        "可回答",
        "建议改为",
        "相关问题",
    ]

    if answer_text and any(marker in answer_text for marker in boundary_markers) or any(marker in answer_lower for marker in ["out of scope", "unsupported", "boundary"]):
        data["boundary_recognition"] = max(clamp_score(data.get("boundary_recognition", 0)), 4)
        data["scope_compliance"] = max(clamp_score(data.get("scope_compliance", 0)), 4)
        data["response_clarity"] = max(clamp_score(data.get("response_clarity", 0)), 3)

    if domain_scope == "out_of_scope" and answer_text:
        data["helpful_redirection"] = max(clamp_score(data.get("helpful_redirection", 0)), 3)
    if answer_text and any(marker in answer_text for marker in redirect_markers):
        data["helpful_redirection"] = max(clamp_score(data.get("helpful_redirection", 0)), 4)

    if domain_scope == "out_of_scope" and not called_tools and not expected_tool:
        data["tool_call_appropriateness"] = max(clamp_score(data.get("tool_call_appropriateness", 0)), 5)

    if domain_scope == "mixed" and expected_tool:
        expected_tools = {item.strip() for item in expected_tool.split("|") if item.strip()}
        if expected_tools and set(called_tools).issubset(expected_tools):
            data["tool_call_appropriateness"] = max(clamp_score(data.get("tool_call_appropriateness", 0)), 4)

    if domain_scope == "out_of_scope":
        data["result_fusion_quality"] = max(clamp_score(data.get("result_fusion_quality", 0)), 4)
        data["answer_optimization_effect"] = max(clamp_score(data.get("answer_optimization_effect", 0)), 4)

    if answer_text:
        data["response_clarity"] = max(clamp_score(data.get("response_clarity", 0)), 3)

    return data


def normalize_standard_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    standard_keys = [
        "retrieval_relevance",
        "answer_accuracy",
        "answer_completeness",
        "reasoning_effectiveness",
        "tool_call_appropriateness",
        "result_fusion_quality",
        "answer_optimization_effect",
    ]
    for key in standard_keys:
        data[key] = clamp_score(data.get(key, 0))

    for key in ["boundary_recognition", "scope_compliance", "response_clarity", "helpful_redirection"]:
        data[key] = clamp_score(data.get(key, 0))

    data["final_score_basis"] = data.get("final_score_basis") or "Average retrieval_relevance, answer_accuracy, answer_completeness"
    return data


def build_fallback_result(error_message: str, evaluation_mode: str = "standard") -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "evaluation_mode": evaluation_mode,
        "suggestion": error_message,
    }
    if evaluation_mode == "boundary":
        data.update(
            {
                "boundary_recognition": 1,
                "scope_compliance": 1,
                "response_clarity": 1,
                "helpful_redirection": 1,
                "tool_call_appropriateness": 1,
            }
        )
        return normalize_boundary_fields(data)
    return normalize_standard_fields(data)


def build_boundary_fallback_from_state(state: EvaluationState, error_message: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "evaluation_mode": "boundary",
        "boundary_recognition": 2,
        "scope_compliance": 2,
        "response_clarity": 2,
        "helpful_redirection": 1,
        "tool_call_appropriateness": 1,
        "suggestion": error_message,
    }
    data = apply_boundary_heuristics(state, data)
    return normalize_boundary_fields(data)


def parse_input_node(state: EvaluationState) -> dict:
    query = state.get("query", "")
    reasoning_result = state.get("reasoning_result", {}) or {}
    evaluation_context = state.get("evaluation_context", {}) or {}

    fused_results = reasoning_result.get("fused_results", [])
    raw_answer = reasoning_result.get("raw_answer", "")
    optimized_answer = reasoning_result.get("optimized_answer", "")
    retrieve_rounds = reasoning_result.get("retrieve_rounds", 0)
    called_tools = reasoning_result.get("called_tools", [])
    tool_results = reasoning_result.get("tool_results", [])
    evaluation_mode = infer_evaluation_mode(evaluation_context)

    print(
        f"[INFO] Evaluation input parsed. mode={evaluation_mode}, "
        f"tools={called_tools}, fused_results={len(fused_results) if isinstance(fused_results, list) else 0}"
    )

    return {
        "query": query,
        "reasoning_result": reasoning_result,
        "evaluation_context": evaluation_context if isinstance(evaluation_context, dict) else {},
        "fused_results": fused_results if isinstance(fused_results, list) else [],
        "raw_answer": raw_answer,
        "optimized_answer": optimized_answer,
        "retrieve_rounds": retrieve_rounds,
        "called_tools": called_tools if isinstance(called_tools, list) else [],
        "tool_results": tool_results if isinstance(tool_results, list) else [],
        "evaluation_mode": evaluation_mode,
        "step": "input_parsed",
        "messages": [AIMessage(content=f"evaluation_mode={evaluation_mode}")],
    }


def format_results_node(state: EvaluationState) -> dict:
    fused_results_str = stringify_results(state.get("fused_results", []))
    tool_results_str = stringify_results(state.get("tool_results", []))
    evaluation_context_str = stringify_context(state.get("evaluation_context", {}))

    return {
        "fused_results_str": fused_results_str,
        "tool_results_str": tool_results_str,
        "evaluation_context_str": evaluation_context_str,
        "step": "formatted",
        "messages": [AIMessage(content="evaluation material formatted")],
    }


def call_llm_evaluation_node(state: EvaluationState) -> dict:
    try:
        evaluation_mode = state.get("evaluation_mode", "standard")
        prompt_template = BOUNDARY_PROMPT_TEMPLATE if evaluation_mode == "boundary" else STANDARD_PROMPT_TEMPLATE

        prompt = prompt_template.format(
            query=state.get("query", ""),
            fused_results=state.get("fused_results_str", "[]"),
            raw_answer=state.get("raw_answer", ""),
            optimized_answer=state.get("optimized_answer", ""),
            retrieve_rounds=state.get("retrieve_rounds", 0),
            called_tools=str(state.get("called_tools", [])),
            tool_results=state.get("tool_results_str", "[]"),
            evaluation_context=state.get("evaluation_context_str", "{}"),
        )

        client = get_openai_client()
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_response = response.choices[0].message.content or "{}"
        print(f"[OK] LLM evaluation finished. mode={evaluation_mode}, chars={len(raw_response)}")
        return {
            "llm_response": raw_response,
            "step": "llm_called",
            "messages": [AIMessage(content="llm evaluation completed")],
        }
    except Exception as exc:
        error_msg = f"LLM evaluation failed: {str(exc)}"
        print(f"[ERROR] {error_msg}")
        return {
            "error": error_msg,
            "step": "llm_call_failed",
            "messages": [AIMessage(content=error_msg)],
        }


def parse_json_node(state: EvaluationState) -> dict:
    raw_response = state.get("llm_response", "")
    if not raw_response:
        return {
            "error": "empty llm response",
            "step": "parse_failed",
            "messages": [AIMessage(content="empty llm response")],
        }

    candidates = [raw_response.strip()]
    cleaned = raw_response.strip()
    cleaned = re.sub(r"```json|```", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned:
        candidates.append(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            return {
                "evaluation_data": data,
                "step": "json_parsed",
                "messages": [AIMessage(content="json parsed")],
            }
        except Exception:
            continue

    error_msg = f"failed to parse evaluation json: {raw_response[:200]}"
    print(f"[WARN] {error_msg}")
    return {
        "error": error_msg,
        "step": "parse_failed",
        "messages": [AIMessage(content=error_msg)],
    }


def validate_result_node(state: EvaluationState) -> dict:
    mode = state.get("evaluation_mode", "standard")
    data = dict(state.get("evaluation_data") or {})
    data["evaluation_mode"] = data.get("evaluation_mode") or mode

    try:
        if data["evaluation_mode"] == "boundary":
            data = apply_boundary_heuristics(state, data)
            normalized = normalize_boundary_fields(data)
        else:
            normalized = normalize_standard_fields(data)

        validated = EvaluationResult(**normalized)
        print(
            f"[OK] Evaluation validated. mode={validated.evaluation_mode}, "
            f"relevance={validated.retrieval_relevance}, accuracy={validated.answer_accuracy}, "
            f"completeness={validated.answer_completeness}"
        )
        return {
            "evaluation_result": validated.model_dump(),
            "step": "validated",
            "messages": [AIMessage(content="evaluation validated")],
        }
    except Exception as exc:
        error_msg = f"validation failed: {str(exc)}"
        if mode == "boundary":
            fallback = build_boundary_fallback_from_state(state, error_msg)
        else:
            fallback = build_fallback_result(error_msg, mode)
        return {
            "evaluation_result": fallback,
            "error": error_msg,
            "step": "validated_with_fallback",
            "messages": [AIMessage(content=error_msg)],
        }


def finalize_node(state: EvaluationState) -> dict:
    return {"step": "completed", "messages": [AIMessage(content="evaluation workflow completed")]}


def error_handler_node(state: EvaluationState) -> dict:
    error = state.get("error", "unknown evaluation error")
    mode = state.get("evaluation_mode", "standard")
    if mode == "boundary":
        fallback = build_boundary_fallback_from_state(state, f"evaluation pipeline error: {error}")
    else:
        fallback = build_fallback_result(f"evaluation pipeline error: {error}", mode)
    return {
        "evaluation_result": fallback,
        "error": error,
        "step": "error_handled",
        "messages": [AIMessage(content=error)],
    }


def after_llm_call(state: EvaluationState) -> str:
    return "error_handler" if state.get("error") else "parse_json"


def after_parse(state: EvaluationState) -> str:
    return "error_handler" if state.get("error") else "validate_result"


def create_evaluation_agent():
    workflow = StateGraph(EvaluationState)
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("format_results", format_results_node)
    workflow.add_node("call_llm", call_llm_evaluation_node)
    workflow.add_node("parse_json", parse_json_node)
    workflow.add_node("validate_result", validate_result_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.add_edge(START, "parse_input")
    workflow.add_edge("parse_input", "format_results")
    workflow.add_edge("format_results", "call_llm")
    workflow.add_conditional_edges("call_llm", after_llm_call, {"parse_json": "parse_json", "error_handler": "error_handler"})
    workflow.add_conditional_edges("parse_json", after_parse, {"validate_result": "validate_result", "error_handler": "error_handler"})
    workflow.add_edge("validate_result", "finalize")
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)


class EvaluationGraphAgent:
    def __init__(self):
        self.app = create_evaluation_agent()

    def evaluate(
        self,
        query: str,
        reasoning_agent_result: Dict[str, Any],
        evaluation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = evaluation_context or {}
        thread_id = f"evaluation-{hash(query)}-{hash(str(reasoning_agent_result))}-{hash(str(context))}"
        inputs = {
            "messages": [HumanMessage(content=f"evaluate: {query[:50]}")],
            "query": query,
            "reasoning_result": reasoning_agent_result,
            "evaluation_context": context,
            "step": "start",
        }
        try:
            result = self.app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
            return {
                "success": not result.get("error"),
                "data": result.get("evaluation_result", {}),
                "step": result.get("step", "completed"),
                "error": result.get("error"),
            }
        except Exception as exc:
            error_msg = f"evaluation workflow failed: {str(exc)}"
            print(f"[ERROR] {error_msg}")
            mode = infer_evaluation_mode(context)
            return {
                "success": False,
                "data": build_fallback_result(error_msg, mode),
                "error": error_msg,
            }


if __name__ == "__main__":
    agent = EvaluationGraphAgent()
    mock = {
        "success": True,
        "fused_results": [],
        "raw_answer": "This question is out of scope for the system.",
        "optimized_answer": "This question is out of scope for the system.",
        "retrieve_rounds": 0,
        "called_tools": [],
        "tool_results": [],
        "error": None,
    }
    res = agent.evaluate(
        "推荐一部电影",
        mock,
        {"id": "O-01", "expected_tool": "", "expected_behavior": "refuse out of scope"},
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
