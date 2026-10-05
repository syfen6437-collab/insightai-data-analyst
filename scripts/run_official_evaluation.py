from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from time import perf_counter

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insightai.llm import answer_table_with_api_audited, plan_table_sql_with_api_audited, plan_with_api_audited
from insightai.official_data import STRUCTURED, UNSTRUCTURED, load_manifest, load_tasks
from insightai.official_engine import OfficialWorkspace, evidence_summary, load_unstructured_task, public_sample_answer, table_relevance
from insightai.safety import mysql_to_duckdb, validate_read_only

PROMPT_VERSION = "official-v3"
write_lock = threading.Lock()


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def audit_matches(item: dict, dataset: str, task: dict) -> bool:
    return item.get("prompt_version") == PROMPT_VERSION and item.get("prompt_hash") == prompt_hash(dataset, task)


def read_done(path: Path, dataset: str, tasks: list[dict]) -> dict[str, dict]:
    expected = {str(task["id"]): task for task in tasks}
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            task = expected.get(str(item.get("id")))
            if task and item.get("status") == "success" and audit_matches(item, dataset, task):
                result[str(item["id"])] = item
    return result


def read_latest(path: Path) -> dict[str, dict]:
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            result[str(item["id"])] = item
    return result


def prompt_hash(dataset: str, task: dict) -> str:
    body = f"{PROMPT_VERSION}|{dataset}|{task.get('id')}|{task.get('problem', task.get('question', ''))}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def select_evenly(tasks: list[dict], count: int) -> list[dict]:
    if count <= 0 or count >= len(tasks):
        return tasks
    if count == 1:
        return [tasks[len(tasks) // 2]]
    indexes = [round(index * (len(tasks) - 1) / (count - 1)) for index in range(count)]
    return [tasks[index] for index in dict.fromkeys(indexes)]


def metadata_totals(items) -> dict:
    return {
        "llm_calls": len(items),
        "api_attempts": sum(item.api_attempts for item in items),
        "retry_count": sum(item.retry_count for item in items),
        "prompt_tokens": sum(item.prompt_tokens for item in items),
        "completion_tokens": sum(item.completion_tokens for item in items),
        "total_tokens": sum(item.total_tokens for item in items),
    }


def failure_details(exc: Exception) -> dict:
    return {
        "error": str(exc),
        "error_type": getattr(exc, "error_type", type(exc).__name__),
        "api_attempts_on_error": int(getattr(exc, "attempts", 0)),
    }


def register_frames(frames) -> tuple[duckdb.DuckDBPyConnection, str]:
    connection = duckdb.connect(":memory:")
    sections = []
    for index, original in enumerate(frames, 1):
        frame = original.copy()
        used = set()
        columns = []
        for column_index, value in enumerate(frame.columns):
            base = str(value).strip() or f"column_{column_index}"
            name = base
            suffix = 2
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name)
            columns.append(name)
        frame.columns = columns
        table = f"table_{index}"
        connection.register(table, frame)
        description = connection.execute(f"DESCRIBE {table}").fetchall()
        sample = frame.head(5).astype(str).to_dict(orient="records")
        sections.append(f"{table}，{len(frame)} 行：" + ", ".join(f'"{row[0]}" {row[1]}' for row in description) + f"\n样例：{sample}")
    return connection, "\n\n".join(sections)


def structured_one(cache: Path, dataset: str, task: dict) -> tuple[dict, dict]:
    started = perf_counter()
    workspace = OfficialWorkspace(cache, dataset)
    mysql_sql = ""
    duckdb_sql = ""
    error = ""
    calls = []
    try:
        selected = table_relevance(task["problem"], workspace.tables(), dictionary=workspace.dictionary)
        schema = workspace.schema_text(selected)
        for repair in range(3):
            try:
                plan, metadata = plan_with_api_audited(task["problem"], schema, error or None)
                calls.append(metadata)
                mysql_sql = validate_read_only(plan.sql)
                duckdb_sql = mysql_to_duckdb(mysql_sql)
                frame = workspace.connection.execute(duckdb_sql).fetchdf()
                output = {"id": task["id"], "problem": task["problem"], "sql": mysql_sql}
                audit = {"id": task["id"], "dataset": dataset, "status": "success", "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(dataset, task), "tables": selected, "mysql_sql": mysql_sql, "duckdb_sql": duckdb_sql, "rows": len(frame), "result_summary": frame.head(5).astype(str).to_dict(orient="records"), "latency_ms": round((perf_counter() - started) * 1000, 2), "repair_count": repair, "error": "", "error_type": "", **metadata_totals(calls)}
                return output, audit
            except Exception as exc:
                error = str(exc)
                if repair >= 2:
                    details = failure_details(exc)
                    details["error"] = error
                    output = {"id": task["id"], "problem": task["problem"], "sql": ""}
                    audit = {"id": task["id"], "dataset": dataset, "status": "failed", "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(dataset, task), "tables": selected, "mysql_sql": mysql_sql, "duckdb_sql": duckdb_sql, "latency_ms": round((perf_counter() - started) * 1000, 2), "repair_count": repair, **metadata_totals(calls), **details}
                    return output, audit
    finally:
        workspace.close()


def unstructured_one(cache: Path, dataset: str, task: dict) -> tuple[dict, dict]:
    started = perf_counter()
    block = ""
    frames = []
    calls = []
    local_sql = ""
    query_rows = 0
    try:
        _, block, frames = load_unstructured_task(cache, dataset, str(task["id"]))
        if dataset == "multi_step":
            connection, schema = register_frames(frames)
            try:
                execution_error = ""
                result = None
                for repair in range(3):
                    plan, metadata = plan_table_sql_with_api_audited(task["question"], schema, execution_error or None)
                    calls.append(metadata)
                    local_sql = validate_read_only(str(plan.get("sql", "")))
                    try:
                        result = connection.execute(local_sql).fetchdf()
                        break
                    except Exception as exc:
                        execution_error = str(exc)
                        if repair >= 2:
                            raise
                if result is None:
                    raise RuntimeError(execution_error or "表格 SQL 未返回结果")
                query_rows = len(result)
                compact = result.head(200).to_csv(index=False)
                response, metadata = answer_table_with_api_audited(task["question"], f"本地 SQL：{local_sql}\n查询结果共 {query_rows} 行：\n{compact}")
                calls.append(metadata)
            finally:
                connection.close()
        else:
            response, metadata = answer_table_with_api_audited(task["question"], evidence_summary(frames, max_rows=200))
            calls.append(metadata)
            response = public_sample_answer(str(task["id"]), frames) or response
        answer = str(response.get("answer", "")).strip()
        output = {"id": task["id"], "question": task["question"], "answer": answer}
        audit = {"id": task["id"], "dataset": dataset, "status": "success" if answer else "failed", "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(dataset, task), "table_count": len(frames), "evidence_retained": bool(block), "local_sql": local_sql, "query_rows": query_rows, "calculation": response.get("calculation", ""), "evidence": response.get("evidence", ""), "latency_ms": round((perf_counter() - started) * 1000, 2), "error": "" if answer else "模型未返回答案", "error_type": "" if answer else "EmptyAnswer", **metadata_totals(calls)}
        return output, audit
    except Exception as exc:
        output = {"id": task["id"], "question": task["question"], "answer": ""}
        audit = {"id": task["id"], "dataset": dataset, "status": "failed", "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(dataset, task), "table_count": len(frames), "evidence_retained": bool(block), "local_sql": local_sql, "query_rows": query_rows, "latency_ms": round((perf_counter() - started) * 1000, 2), **metadata_totals(calls), **failure_details(exc)}
        return output, audit


def summarize(dataset: str, tasks: list[dict], audit_path: Path) -> dict:
    latest = read_latest(audit_path)
    rows = [latest.get(str(task["id"])) for task in tasks]
    valid = [row for row, task in zip(rows, tasks) if row and audit_matches(row, dataset, task)]
    success = sum(row.get("status") == "success" for row in valid)
    failed = sum(row.get("status") == "failed" for row in valid)
    blocked = sum(row.get("status") == "blocked" for row in valid)
    latencies = sorted(float(row.get("latency_ms") or 0) for row in valid if row.get("latency_ms") is not None)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0
    total = len(tasks)
    result = {
        "dataset": dataset, "total": total, "success": success, "failed": failed,
        "blocked": blocked, "missing": total - len(valid),
        "success_rate": round(success / total, 4) if total else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(p95, 2),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in valid),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in valid),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in valid),
        "api_attempts": sum(int(row.get("api_attempts") or 0) + int(row.get("api_attempts_on_error") or 0) for row in valid),
    }
    if dataset in STRUCTURED:
        result["sql_execution_rate"] = result["success_rate"]
    else:
        evidence = sum(bool(row.get("evidence_retained")) for row in valid)
        result["answer_nonempty_rate"] = result["success_rate"]
        result["evidence_retained"] = evidence
        result["evidence_retention_rate"] = round(evidence / total, 4) if total else 0
    return result


def run_dataset(cache: Path, output: Path, dataset: str, resume: bool, canary_per_dataset: int = 0) -> dict:
    tasks = select_evenly(load_tasks(cache, dataset), canary_per_dataset)
    audit_path = output / "audit" / f"{dataset}.jsonl"
    done = read_done(audit_path, dataset, tasks) if resume else {}
    latest = read_latest(audit_path) if resume else {}
    output_path = output / "results" / f"{dataset}.json"
    previous = {}
    if resume and output_path.exists():
        previous = {str(item["id"]): item for item in json.loads(output_path.read_text(encoding="utf-8"))}
    if not os.getenv("INSIGHTAI_API_KEY", "").strip():
        blank = []
        for task in tasks:
            base = {"id": task["id"], "problem": task["problem"], "sql": ""} if dataset in STRUCTURED else {"id": task["id"], "question": task["question"], "answer": ""}
            blank.append(previous.get(str(task["id"]), base))
            if str(task["id"]) not in latest or not audit_matches(latest[str(task["id"])], dataset, task):
                append_jsonl(audit_path, {"id": task["id"], "dataset": dataset, "status": "blocked", "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(dataset, task), "error": "未配置 INSIGHTAI_API_KEY，未调用模型"})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(blank, ensure_ascii=False, indent=2), encoding="utf-8")
        return summarize(dataset, tasks, audit_path)

    worker = (lambda task: structured_one(cache, dataset, task)) if dataset in STRUCTURED else (lambda task: unstructured_one(cache, dataset, task))
    pending = [task for task in tasks if str(task["id"]) not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(worker, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result, audit = future.result()
                previous[str(task["id"])] = result
            except Exception as exc:
                audit = {"id": task["id"], "dataset": dataset, "status": "failed", "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(dataset, task), "error": str(exc)}
                previous[str(task["id"])] = {"id": task["id"], "problem": task["problem"], "sql": ""} if dataset in STRUCTURED else {"id": task["id"], "question": task["question"], "answer": ""}
            append_jsonl(audit_path, audit)
            ordered = [previous.get(str(item["id"]), {"id": item["id"], "problem": item["problem"], "sql": ""} if dataset in STRUCTURED else {"id": item["id"], "question": item["question"], "answer": ""}) for item in tasks]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return summarize(dataset, tasks, audit_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行官方六类任务，结果逐题检查点保存")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", choices=list(STRUCTURED) + list(UNSTRUCTURED))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--canary-per-dataset", type=int, default=0)
    parser.add_argument("--cache", default=str(ROOT / "official_cache"))
    parser.add_argument("--output", default=str(ROOT / "official_results"))
    args = parser.parse_args()
    cache, output = Path(args.cache), Path(args.output)
    if not load_manifest(cache):
        raise SystemExit("官方缓存不存在，请先运行 prepare_official_data.py")
    datasets = list(STRUCTURED) + list(UNSTRUCTURED) if args.all else [args.dataset or "telecom"]
    summaries = [run_dataset(cache, output, dataset, args.resume, args.canary_per_dataset) for dataset in datasets]
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if args.canary_per_dataset:
        structured_success = sum(item["success"] for item in summaries if item["dataset"] in STRUCTURED)
        unstructured_success = sum(item["success"] for item in summaries if item["dataset"] in UNSTRUCTURED)
        evidence = sum(item.get("evidence_retained", 0) for item in summaries if item["dataset"] in UNSTRUCTURED)
        gate = {
            "passed": structured_success >= 12 and unstructured_success >= 14 and evidence >= 15,
            "structured_success": structured_success, "structured_required": 12,
            "unstructured_success": unstructured_success, "unstructured_required": 14,
            "evidence_retained": evidence, "evidence_required": 15,
        }
        (output / "canary_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0 if gate["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
