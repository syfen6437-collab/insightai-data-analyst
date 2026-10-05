from __future__ import annotations

import html
from datetime import datetime

from .analysis import QueryResult, summarize


def markdown_report(history: list[QueryResult]) -> str:
    lines = ["# 数问 InsightAI 分析报告", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", ""]
    for index, item in enumerate(history, 1):
        lines.extend([f"## {index} {item.question}", "", f"状态：{item.status}", ""])
        if item.plan:
            finding, suggestions = summarize(item.data, item.plan.title)
            lines.extend([f"### {item.plan.title}", "", finding, "", "```sql", item.plan.sql, "```", "", "建议："])
            lines.extend(f"- {suggestion}" for suggestion in suggestions)
            lines.extend(["", "数据口径：结果仅基于当前上传数据。", ""])
        else:
            lines.extend([item.message, ""])
    return "\n".join(lines)


def html_report(history: list[QueryResult]) -> str:
    body = markdown_report(history)
    escaped = html.escape(body)
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>数问 InsightAI 分析报告</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:920px;margin:48px auto;color:#17221d;line-height:1.7}}pre{{white-space:pre-wrap;background:#f4f6f5;padding:24px;border-left:4px solid #177245}}</style></head><body><pre>{escaped}</pre></body></html>"""
