from __future__ import annotations

import re

import sqlglot
from sqlglot import exp


class UnsafeQueryError(ValueError):
    """Raised when a generated statement is not a read-only query."""


FORBIDDEN = {
    "alter",
    "attach",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "install",
    "load",
    "merge",
    "pragma",
    "replace",
    "truncate",
    "update",
    "vacuum",
}


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\r\n]*", " ", sql)


def validate_read_only(sql: str) -> str:
    """Parse one statement and reject every non-query AST node."""
    if not sql or not sql.strip():
        raise UnsafeQueryError("SQL 为空")
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if ";" in cleaned:
        raise UnsafeQueryError("仅允许执行一条查询语句")
    try:
        statements = sqlglot.parse(cleaned, read="mysql")
    except sqlglot.errors.ParseError as exc:
        raise UnsafeQueryError(f"SQL 解析失败：{exc}") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise UnsafeQueryError("仅允许一条 SELECT 或只读 CTE 查询")
    blocked_nodes = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
    if any(statements[0].find(node) is not None for node in blocked_nodes):
        raise UnsafeQueryError("检测到写入或结构变更操作")
    words = set(re.findall(r"\b[a-zA-Z_]+\b", cleaned.lower()))
    blocked = sorted(words & FORBIDDEN)
    if blocked:
        raise UnsafeQueryError(f"检测到危险关键字：{', '.join(blocked)}")
    return cleaned


def enforce_limit(sql: str, limit: int = 500) -> str:
    cleaned = validate_read_only(sql)
    if re.search(r"\blimit\s+\d+\b", cleaned, flags=re.I):
        return cleaned
    return f"{cleaned}\nLIMIT {int(limit)}"


def mysql_to_duckdb(sql: str) -> str:
    cleaned = validate_read_only(sql)
    try:
        translated = sqlglot.transpile(cleaned, read="mysql", write="duckdb", pretty=True)[0]
        # sqlglot represents MySQL JSON_UNQUOTE(JSON_EXTRACT(json, '$.path'))
        # as JSON_UNQUOTE(json -> '$.path'). DuckDB exposes the unquoted form
        # through the ->> operator, so normalize only this read-only pattern.
        translated = re.sub(
            r"JSON_UNQUOTE\(\s*([^()]+?)\s*->\s*('(?:''|[^'])*')\s*\)",
            r"\1 ->> \2",
            translated,
            flags=re.IGNORECASE,
        )
        # DuckDB's JSON arrow operator returns JSON, which cannot be compared
        # directly with a VARCHAR literal. Normalize both arrow forms to a
        # string extraction so filters such as city = '北京市' work reliably.
        translated = re.sub(
            r"([A-Za-z_][A-Za-z0-9_.]*)\s*->>?\s*('(?:''|[^'])*')",
            r"json_extract_string(\1, \2)",
            translated,
            flags=re.IGNORECASE,
        )
        return translated
    except sqlglot.errors.SqlglotError as exc:
        raise UnsafeQueryError(f"SQL 方言转换失败：{exc}") from exc


def preview_sql(sql: str, limit: int = 500) -> str:
    """Limit only the UI execution copy, never the submitted SQL."""
    cleaned = validate_read_only(sql)
    return f"SELECT * FROM ({cleaned}) AS insightai_preview LIMIT {int(limit)}"
