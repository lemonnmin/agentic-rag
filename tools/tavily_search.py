#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   tavily_search.py
@Time    :   2026/03/10 22:38:09
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   可供调用的搜索接口工具
"""
import builtins
import os
import sys
from tavily import TavilyClient
from dotenv import load_dotenv


def safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sanitized_args = []
        for arg in args:
            text = str(arg)
            sanitized_args.append(text.encode(encoding, errors="ignore").decode(encoding, errors="ignore"))
        builtins.print(*sanitized_args, **kwargs)


print = safe_print

def load_env_from_root():
    """向上遍历目录，找到项目根目录的.env文件并加载"""
    # 从当前脚本目录开始向上找
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 最多向上找5级，避免无限循环
    for _ in range(5):
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print(f"✅ 找到.env文件：{env_path}")
            return
        # 向上一级目录
        current_dir = os.path.dirname(current_dir)
    print("❌ 未找到.env文件")

# 调用函数加载.env
load_env_from_root()

class TavilySearchTool:
    """
    Tavily 搜索工具
    """

    name = "web_search"
    description = "用于搜索互联网信息"

    def __init__(self, api_key=None):

        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError("TAVILY_API_KEY 未设置")

        self.client = TavilyClient(api_key=api_key)

    def run(self, query: str, max_results: int = 3):
        """
        Agent 调用入口
        """

        try:

            response = self.client.search(
                query=query,
                max_results=max_results
            )

            results = response.get("results", [])

            data = []

            for i, r in enumerate(results, start=1):

                data.append({
                    "rank": i,
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "source": "tavily"
                })

            return {
                "tool": self.name,
                "success": True,
                "query": query,
                "count": len(data),
                "data": data,
                "error": None
            }

        except Exception as e:

            return {
                "tool": self.name,
                "success": False,
                "query": query,
                "count": 0,
                "data": [],
                "error": str(e)
            }

if __name__ == "__main__":

    tool = TavilySearchTool()

    result = tool.run("香菇种植条件")

    print(result)
