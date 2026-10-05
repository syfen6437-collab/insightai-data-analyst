from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from time import perf_counter, sleep

from dotenv import load_dotenv

from .offline import QueryPlan


load_dotenv()

SYSTEM_PROMPT = """你是数据分析系统的 Text2SQL 模块。只能生成 MySQL 8 兼容的只读 SQL。
禁止写入、删除、建表、PRAGMA、ATTACH 和多语句。只使用给定表和字段。
无法回答时将 sql 设为空字符串，并在 explanation 中说明缺少字段。
输出严格 JSON，字段为 intent、sql、chart_type、title、explanation。
chart_type 只能是 metric、table、line、bar、pie、scatter。"""

TABLE_QA_PROMPT = """你是表格问答模块。仅依据给定表格证据回答，不得补充外部事实。
输出严格 JSON：answer 为规范化最终答案，calculation 为简短计算过程，evidence 为使用的行列。
百分比保留题目所需精度。题目要求预测、增长率或统计量时，必须选择合理且可复核的方法完成计算并在 calculation 中说明。
如果问题比较两个年份，而表格行中已经提供这两个年份及对应数值，直接按行值计算变化率
（后一年 - 前一年）/ 前一年 * 100%，不要把表头中出现的业务年份误认为缺失数据。
表格只有两行日期和一个数值时，优先使用这两行完成题目要求，并在 calculation 中写出代入的数字。
answer 不得为空；没有符合条件的记录时回答“无符合条件项”，确实无法由证据确定时回答“无法根据给定表格确定”并说明缺少的信息。"""

TABLE_SQL_PROMPT = """你是表格计算规划模块。请把问题转换为一条只读 DuckDB SQL。
只能使用给定的 table_1、table_2 等临时表及其字段，只能输出 SELECT 或只读 CTE，禁止写入、DDL、多语句和环境操作。
SQL 必须完成筛选、分组、排序、增长率、中位数、复合增长率或其他所需计算，尽量返回最终候选与关键数值。
列名包含空格、中文或特殊字符时使用双引号。输出严格 JSON：sql、explanation。"""


@dataclass(frozen=True)
class LLMCallMetadata:
    latency_ms: float
    retry_count: int
    api_attempts: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMRequestError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, error_type: str):
        super().__init__(message)
        self.attempts = attempts
        self.error_type = error_type


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON")
    return json.loads(text[start : end + 1])


def _chat_json(system: str, user: str, retries: int = 3) -> tuple[dict, LLMCallMetadata]:
    api_key = os.getenv("INSIGHTAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 INSIGHTAI_API_KEY")
    base_url = os.getenv("INSIGHTAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("INSIGHTAI_MODEL", "deepseek-chat")
    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    started = perf_counter()
    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(f"{base_url}/chat/completions", data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            usage = body.get("usage") or {}
            metadata = LLMCallMetadata(
                latency_ms=(perf_counter() - started) * 1000,
                retry_count=attempt,
                api_attempts=attempt + 1,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
            )
            return _extract_json(body["choices"][0]["message"]["content"]), metadata
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep(2 ** attempt)
    raise LLMRequestError(
        f"模型接口调用失败：{last_error}",
        attempts=retries,
        error_type=type(last_error).__name__ if last_error else "UnknownError",
    )


def plan_with_api_audited(question: str, schema: str, error: str | None = None) -> tuple[QueryPlan, LLMCallMetadata]:
    user = f"数据库结构：\n{schema}\n\n用户问题：{question}"
    if error:
        user += f"\n\n上一次 SQL 执行错误：{error}\n请修复 SQL。"
    data, metadata = _chat_json(SYSTEM_PROMPT, user)
    plan = QueryPlan(
        str(data.get("intent", "query")),
        str(data.get("sql", "")),
        str(data.get("chart_type", "table")),
        str(data.get("title", "查询结果")),
        str(data.get("explanation", "")),
    )
    return plan, metadata


def plan_with_api(question: str, schema: str, error: str | None = None) -> QueryPlan:
    return plan_with_api_audited(question, schema, error)[0]


def answer_table_with_api_audited(question: str, evidence: str) -> tuple[dict, LLMCallMetadata]:
    return _chat_json(TABLE_QA_PROMPT, f"问题：{question}\n\n表格证据：\n{evidence}")


def plan_table_sql_with_api_audited(question: str, schema: str, error: str | None = None) -> tuple[dict, LLMCallMetadata]:
    user = f"临时表结构与样例：\n{schema}\n\n问题：{question}"
    if error:
        user += f"\n\n上一次 SQL 执行错误：{error}\n请修复 SQL。"
    return _chat_json(TABLE_SQL_PROMPT, user)


def answer_table_with_api(question: str, evidence: str) -> dict:
    data, metadata = answer_table_with_api_audited(question, evidence)
    data["latency_ms"] = metadata.latency_ms
    data["retries"] = metadata.retry_count
    return data
