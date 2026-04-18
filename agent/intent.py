#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   intent.py
@Time    :   2026/03/15
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   意图解析代理：技能化实现
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


def load_env_from_root() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            return
        current_dir = os.path.dirname(current_dir)


load_env_from_root()


class IntentResult(BaseModel):
    intent_type: str = Field(description="rag_search/web_search/weather/unknown")
    domain: str = Field(description="食用菌子领域：香菇/金针菇/杏鲍菇/其他食用菌")
    keywords: List[str] = Field(description="核心关键词列表")
    complexity: str = Field(description="问题复杂度：simple/complex/multi_turn")
    sub_tasks: List[str] = Field(description="拆解后的子任务列表")
    city: Optional[str] = Field(default=None, description="关联城市（天气相关）")


class Skill:
    def __init__(self, name: str, handler: Callable[[Any], Dict[str, Any]]) -> None:
        self.name = name
        self.handler = handler

    def execute(self, payload: Any) -> Dict[str, Any]:
        return self.handler(payload)


class Agent:
    def __init__(self, skills: Dict[str, Skill]) -> None:
        self.skills = skills

    def perceive(self, raw_input: str) -> Dict[str, Any]:
        return {
            "raw_input": raw_input.strip(),
            "tokens": raw_input.split(),
            "is_question": raw_input.strip().endswith("?"),
            "keywords": [word.lower() for word in raw_input.split() if len(word) > 3],
        }

    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not observation["raw_input"]:
            return {"action": "fallback", "reason": "empty input", "payload": ""}

        return {"action": "use_skill", "skill": "parse_intent", "payload": observation["raw_input"]}

    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        action = plan.get("action")
        if action == "use_skill":
            skill_name = plan.get("skill")
            skill = self.skills.get(skill_name)
            if not skill:
                return {"status": "error", "message": f"未找到技能 {skill_name}"}
            return {"status": "completed", "action": action, "result": skill.execute(plan["payload"])}

        if action == "fallback":
            fallback_skill = self.skills.get("fallback")
            if not fallback_skill:
                return {"status": "error", "message": "未配置兜底技能"}
            return {
                "status": "completed",
                "action": action,
                "result": fallback_skill.execute({"query": plan.get("payload", ""), "reason": plan.get("reason", "empty input")}),
            }

        return {"status": "error", "message": "未识别的请求动作。"}

    def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "result": result,
            "summary": f"Action={result.get('action')} status={result.get('status')}"
        }

    def run(self, raw_input: str) -> Dict[str, Any]:
        observation = self.perceive(raw_input)
        plan = self.plan(observation)
        result = self.act(plan)
        return self.reflect(result)


class IntentParser:
    def __init__(self) -> None:
        self.intent_keywords = {
            "rag_search": ["种植", "温度", "湿度", "培养基", "周期", "通风", "配方"],
            "web_search": ["最新", "市场价格", "政策", "2025", "新技术", "行情"],
            "weather": ["天气", "气温", "降雨", "寒潮", "高温", "气候"],
        }
        self.domain_keywords = {
            "香菇": ["香菇", "香蕈", "冬菇"],
            "金针菇": ["金针菇", "毛柄金钱菌"],
            "杏鲍菇": ["杏鲍菇", "刺芹侧耳"],
        }
        self.complexity_keywords = {
            "complex": ["哪些", "多少", "如何控制", "怎么调节", "条件"],
            "multi_turn": ["为什么", "怎么办", "如何解决", "多次", "步骤"],
        }

    def extract_city(self, query: str) -> Optional[str]:
        city_pattern = r"北京|上海|广州|深圳|杭州|成都|重庆|南京|武汉|西安|郑州|济南|青岛"
        match = re.search(city_pattern, query)
        return match.group() if match else None

    def _extract_intent_by_rule(self, query: str) -> str:
        for intent, keys in self.intent_keywords.items():
            if any(key in query for key in keys):
                return intent
        return "unknown"

    def _extract_domain_by_rule(self, query: str) -> str:
        for domain, keys in self.domain_keywords.items():
            if any(key in query for key in keys):
                return domain
        return "其他食用菌"

    def _extract_keywords_by_rule(self, query: str) -> List[str]:
        city = self.extract_city(query)
        if city:
            query = query.replace(city, "")

        keyword_pattern = r"香菇|金针菇|杏鲍菇|温度|湿度|种植|周期|通风|培养基|天气|气候"
        keywords = re.findall(keyword_pattern, query)
        if len(keywords) < 2:
            fallback = re.split(r"的|需要|控制|哪些|和", query)
            keywords = [k.strip() for k in fallback if k.strip()][:2]
        return keywords[:5]

    def _extract_complexity_by_rule(self, query: str) -> str:
        if any(key in query for key in self.complexity_keywords["multi_turn"]):
            return "multi_turn"
        if any(key in query for key in self.complexity_keywords["complex"]):
            return "complex"
        return "simple"

    def _split_sub_tasks(self, query: str, complexity: str) -> List[str]:
        if complexity == "simple":
            return []
        tasks: List[str] = []
        if "温度" in query and "湿度" in query:
            city = self.extract_city(query) or ""
            domain = self._extract_domain_by_rule(query)
            tasks.append(f"{city}{domain}种植适宜温度")
            tasks.append(f"{city}{domain}种植适宜湿度")
        return tasks[:3]

    def rule_parse(self, query: str) -> Dict[str, Any]:
        intent_type = self._extract_intent_by_rule(query)
        domain = self._extract_domain_by_rule(query)
        keywords = self._extract_keywords_by_rule(query)
        complexity = self._extract_complexity_by_rule(query)
        sub_tasks = self._split_sub_tasks(query, complexity)
        city = self.extract_city(query)

        return {
            "intent_type": intent_type,
            "domain": domain,
            "keywords": keywords,
            "complexity": complexity,
            "sub_tasks": sub_tasks,
            "city": city,
        }

    def llm_parse(self, query: str) -> Dict[str, Any]:
        prompt = f"""你是一个专业的意图解析助手。请分析用户的问题并返回JSON格式的解析结果。

用户问题：{query}

必须包含以下字段：
- intent_type: rag_search/web_search/weather/unknown
- domain: 香菇/金针菇/杏鲍菇/其他食用菌
- keywords: 核心关键词列表（最多5个）
- complexity: simple/complex/multi_turn
- sub_tasks: 拆解后的子任务列表（如果没有子任务则返回空列表）
- city: 关联城市（如果没有则返回null）

请确保返回合法的JSON格式。"""
        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            response_text = response.content
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise ValueError("未从LLM响应中提取到JSON")

            json_str = json_match.group()
            json_str = re.sub(r"//.*?\n", "", json_str)
            json_str = re.sub(r",\s*}", "}", json_str)
            llm_result = json.loads(json_str)

            for field in ["intent_type", "domain", "keywords", "complexity", "sub_tasks", "city"]:
                if field not in llm_result:
                    llm_result[field] = None if field == "city" else ([] if field in ["keywords", "sub_tasks"] else "unknown")

            return {"llm_based_result": llm_result, "error": None}
        except Exception as exc:
            return {"llm_based_result": None, "error": f"LLM解析失败：{str(exc)}"}

    def merge(self, rule_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]], llm_error: Optional[str]) -> Dict[str, Any]:
        if llm_result and not llm_error:
            if llm_result.get("intent_type") in ["rag_search", "web_search", "weather", "unknown"]:
                final_intent = llm_result
            else:
                final_intent = rule_result
        else:
            final_intent = rule_result

        if "city" not in final_intent or final_intent["city"] is None:
            final_intent["city"] = rule_result.get("city") or self.extract_city("")

        return {"final_intent": final_intent, "merge_error": llm_error}

    def validate(self, query: str, final_intent: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated_result = IntentResult(**final_intent)
            return {"final_intent": validated_result.model_dump(), "error": None}
        except Exception as exc:
            fallback = IntentResult(
                intent_type="rag_search",
                domain=self._extract_domain_by_rule(query),
                keywords=self._extract_keywords_by_rule(query),
                complexity="simple",
                sub_tasks=[],
                city=self.extract_city(query),
            )
            return {"final_intent": fallback.model_dump(), "error": f"验证失败，使用兜底结果：{str(exc)}"}

    def fallback(self, query: str, reason: str) -> Dict[str, Any]:
        fallback_intent = IntentResult(
            intent_type="rag_search",
            domain=self._extract_domain_by_rule(query),
            keywords=self._extract_keywords_by_rule(query) or ["查询"],
            complexity="simple",
            sub_tasks=[],
            city=self.extract_city(query),
        )
        return {"final_intent": fallback_intent.model_dump(), "error": reason}


llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-32B-Instruct"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0.1,
)


def rule_parser_skill(payload: Any) -> Dict[str, Any]:
    parser = IntentParser()
    query = payload if isinstance(payload, str) else str(payload)
    return parser.rule_parse(query)


def llm_parser_skill(payload: Any) -> Dict[str, Any]:
    parser = IntentParser()
    query = payload if isinstance(payload, str) else str(payload)
    return parser.llm_parse(query)


def merge_intent_skill(payload: Any) -> Dict[str, Any]:
    parser = IntentParser()
    return parser.merge(
        rule_result=payload.get("rule_result", {}),
        llm_result=payload.get("llm_result"),
        llm_error=payload.get("llm_error"),
    )


def validate_intent_skill(payload: Any) -> Dict[str, Any]:
    parser = IntentParser()
    query = payload.get("query", "")
    final_intent = payload.get("final_intent", {})
    return parser.validate(query, final_intent)


def fallback_skill(payload: Any) -> Dict[str, Any]:
    parser = IntentParser()
    query = payload.get("query", "")
    reason = payload.get("reason", "fallback")
    return parser.fallback(query, reason)


def parse_intent_skill(payload: Any) -> Dict[str, Any]:
    parser = IntentParser()
    query = payload if isinstance(payload, str) else str(payload)

    rule_result = parser.rule_parse(query)
    llm_result = None
    llm_error = None

    if rule_result["intent_type"] == "unknown" or not rule_result["keywords"]:
        llm_output = parser.llm_parse(query)
        llm_result = llm_output.get("llm_based_result")
        llm_error = llm_output.get("error")

    merged = parser.merge(rule_result, llm_result, llm_error)
    validated = parser.validate(query, merged["final_intent"])

    return {
        "rule_result": rule_result,
        "llm_result": llm_result,
        "llm_error": llm_error,
        "merged": merged,
        "final_intent": validated["final_intent"],
        "error": validated["error"],
    }


def create_default_agent() -> Agent:
    skills = {
        "rule_parser": Skill("rule_parser", rule_parser_skill),
        "llm_parser": Skill("llm_parser", llm_parser_skill),
        "merge": Skill("merge", merge_intent_skill),
        "validate": Skill("validate", validate_intent_skill),
        "fallback": Skill("fallback", fallback_skill),
        "parse_intent": Skill("parse_intent", parse_intent_skill),
    }
    return Agent(skills)


def parse_intent(query: str) -> dict:
    agent = create_default_agent()
    result = agent.run(query)
    if result["result"].get("status") == "error":
        return {"success": False, "data": {}, "error": result["result"].get("message")}

    final_payload = result["result"].get("result", {})
    return {
        "success": True,
        "data": final_payload.get("final_intent", {}),
        "error": final_payload.get("error"),
        "debug": {
            "rule_result": final_payload.get("rule_result"),
            "llm_result": final_payload.get("llm_result"),
            "merged": final_payload.get("merged"),
        },
    }


if __name__ == "__main__":
    query = "北京种植香菇需要控制哪些温度和湿度条件？"
    output = parse_intent(query)
    print(json.dumps(output, ensure_ascii=False, indent=2))
