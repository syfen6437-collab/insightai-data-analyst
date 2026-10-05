from __future__ import annotations

import io
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: int
    fields: list[dict]


def safe_identifier(value: str, fallback: str = "uploaded_data") -> str:
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", value).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"t_{text}"
    return text[:60]


class DataWorkspace:
    def __init__(self, database_path: str | Path | None = None):
        if database_path:
            self.path = Path(database_path)
            self._temporary = False
        else:
            handle, name = tempfile.mkstemp(prefix="insightai_", suffix=".sqlite")
            os.close(handle)
            self.path = Path(name)
            self._temporary = True
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def close(self) -> None:
        self.connection.close()
        if self._temporary and self.path.exists():
            self.path.unlink(missing_ok=True)

    def load_csv(self, content: bytes, filename: str) -> list[str]:
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                frame = pd.read_csv(io.BytesIO(content), encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"CSV 编码无法识别：{last_error}")
        table = safe_identifier(Path(filename).stem)
        frame.to_sql(table, self.connection, if_exists="replace", index=False)
        return [table]

    def load_excel(self, content: bytes) -> list[str]:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
        loaded = []
        for sheet_name, frame in sheets.items():
            table = safe_identifier(sheet_name, "sheet")
            frame.to_sql(table, self.connection, if_exists="replace", index=False)
            loaded.append(table)
        return loaded

    def load_sqlite(self, content: bytes) -> list[str]:
        handle, name = tempfile.mkstemp(prefix="insightai_upload_", suffix=".sqlite")
        os.close(handle)
        uploaded = Path(name)
        uploaded.write_bytes(content)
        try:
            source = sqlite3.connect(uploaded)
            names = [row[0] for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )]
            for name in names:
                frame = pd.read_sql_query(f'SELECT * FROM "{name}"', source)
                frame.to_sql(safe_identifier(name), self.connection, if_exists="replace", index=False)
            source.close()
            return [safe_identifier(name) for name in names]
        finally:
            uploaded.unlink(missing_ok=True)

    def load_file(self, content: bytes, filename: str) -> list[str]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            return self.load_csv(content, filename)
        if suffix in {".xlsx", ".xls"}:
            return self.load_excel(content)
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            return self.load_sqlite(content)
        raise ValueError("仅支持 CSV、XLSX、XLS、SQLite 和 DB 文件")

    def tables(self) -> list[str]:
        return [row[0] for row in self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]

    def profile(self, table: str) -> TableProfile:
        quoted = table.replace('"', '""')
        frame = pd.read_sql_query(f'SELECT * FROM "{quoted}"', self.connection)
        fields = []
        for column in frame.columns:
            series = frame[column]
            examples = [str(value)[:40] for value in series.dropna().head(3).tolist()]
            fields.append({
                "字段": column,
                "类型": str(series.dtype),
                "空值数": int(series.isna().sum()),
                "唯一值": int(series.nunique(dropna=True)),
                "样例": " / ".join(examples),
            })
        return TableProfile(table, len(frame), len(frame.columns), fields)

    def schema_text(self) -> str:
        parts = []
        for table in self.tables():
            profile = self.profile(table)
            columns = ", ".join(f'{field["字段"]} ({field["类型"]})' for field in profile.fields)
            parts.append(f"表 {table}，{profile.rows} 行：{columns}")
        return "\n".join(parts)

    def execute(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.connection)
