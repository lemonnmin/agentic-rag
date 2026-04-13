#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   base_tool.py
@Time    :   2026/03/11 17:18:10
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   工具基类
"""

class BaseTool:

    name = "base_tool"
    description = ""

    def run(self, **kwargs):
        raise NotImplementedError

    def format_result(self, query, results):

        return {
            "tool": self.name,
            "success": True,
            "query": query,
            "count": len(results),
            "data": results,
            "error": None
        }

    def format_error(self, query, error):

        return {
            "tool": self.name,
            "success": False,
            "query": query,
            "count": 0,
            "data": [],
            "error": str(error)
        }