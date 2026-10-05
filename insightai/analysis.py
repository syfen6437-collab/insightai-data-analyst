from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from .data import DataWorkspace
from .llm import plan_with_api
from .offline import QueryPlan, plan_sample_query
from .safety import UnsafeQueryError, enforce_limit


@dataclass
class QueryResult:
    question: str
    plan: QueryPlan | None
    data: pd.DataFrame
    status: str
    message: str
    elapsed_ms: float


def execute_question(workspace: DataWorkspace, question: str, prefer_api: bool = False) -> QueryResult:
    started = perf_counter()
    try:
        plan = plan_with_api(question, workspace.schema_text()) if prefer_api else plan_sample_query(question)
        if plan is None:
            if prefer_api:
                raise ValueError("模型未生成可执行查询")
            return QueryResult(question, None, pd.DataFrame(), "无法回答", "当前数据缺少回答该问题所需的字段，或离线模式尚未覆盖该问法。", (perf_counter() - started) * 1000)
        sql = enforce_limit(plan.sql)
        try:
            frame = workspace.execute(sql)
        except Exception as first_error:
            if not prefer_api:
                raise
            repaired = plan_with_api(question, workspace.schema_text(), str(first_error))
            sql = enforce_limit(repaired.sql)
            frame = workspace.execute(sql)
            plan = repaired
        plan.sql = sql
        return QueryResult(question, plan, frame, "成功", "查询已完成", (perf_counter() - started) * 1000)
    except UnsafeQueryError as exc:
        return QueryResult(question, plan if 'plan' in locals() else None, pd.DataFrame(), "已拦截", str(exc), (perf_counter() - started) * 1000)
    except Exception as exc:
        return QueryResult(question, plan if 'plan' in locals() else None, pd.DataFrame(), "失败", str(exc), (perf_counter() - started) * 1000)


def summarize(frame: pd.DataFrame, title: str) -> tuple[str, list[str]]:
    if frame.empty:
        return "未返回数据。", ["检查提问是否与当前数据字段匹配。"]
    if frame.shape == (1, 1):
        value = frame.iloc[0, 0]
        return f"{title}为 {value:,.2f}" if isinstance(value, float) else f"{title}为 {value}", ["可继续按时间、区域或品类拆分该指标。"]
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        metric = numeric.columns[0]
        top_index = numeric[metric].idxmax()
        label_col = next((column for column in frame.columns if column not in numeric.columns), None)
        if label_col is not None:
            label = frame.loc[top_index, label_col]
            value = frame.loc[top_index, metric]
            return f"{label} 的 {metric} 最高，为 {value:,.2f}", ["关注最高项与其他项目的差距。", "结合业务口径复核异常值。"]
    return f"查询返回 {len(frame)} 行、{len(frame.columns)} 列结果。", ["可下载结果并继续追问。"]
