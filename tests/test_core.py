from pathlib import Path

from insightai.analysis import execute_question
from insightai.data import DataWorkspace
from insightai.safety import UnsafeQueryError, enforce_limit, validate_read_only
from insightai.official_data import parse_table_block
from scripts.run_official_evaluation import PROMPT_VERSION, prompt_hash, read_done, read_latest, select_evenly


ROOT = Path(__file__).resolve().parents[1]


def test_read_only_validation():
    assert validate_read_only("SELECT 1") == "SELECT 1"
    assert enforce_limit("WITH x AS (SELECT 1 AS n) SELECT * FROM x").endswith("LIMIT 500")


def test_dangerous_sql_rejected():
    for sql in ["DROP TABLE orders", "DELETE FROM orders", "SELECT 1; SELECT 2", "PRAGMA table_info(orders)"]:
        try:
            validate_read_only(sql)
        except UnsafeQueryError:
            pass
        else:
            raise AssertionError(sql)


def test_offline_question():
    workspace = DataWorkspace(ROOT / "sample_data" / "insightai_demo.sqlite")
    result = execute_question(workspace, "各区域销售额从高到低排序")
    assert result.status == "成功"
    assert list(result.data.columns) == ["region", "sales"]
    assert len(result.data) == 5
    workspace.close()


def test_invalid_question():
    workspace = DataWorkspace(ROOT / "sample_data" / "insightai_demo.sqlite")
    result = execute_question(workspace, "员工年龄分布如何")
    assert result.status == "无法回答"
    workspace.close()


def test_markdown_table_with_blank_first_header():
    frames = parse_table_block("""```markdown
||amount|
|---|---|
|2015 net revenue|2408.8|
```""")
    assert len(frames) == 1
    assert frames[0].shape == (1, 2)


def test_latest_audit_status_wins(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"id":1,"status":"blocked"}\n{"id":1,"status":"success"}\n', encoding="utf-8")
    assert read_latest(path)["1"]["status"] == "success"


def test_resume_requires_current_prompt(tmp_path):
    tasks = [{"id": 1, "problem": "test"}]
    path = tmp_path / "audit.jsonl"
    path.write_text(
        '{"id":1,"status":"success","prompt_version":"old","prompt_hash":"old"}\n'
        + '{"id":1,"status":"success","prompt_version":"'
        + PROMPT_VERSION
        + '","prompt_hash":"'
        + prompt_hash("medical", tasks[0])
        + '"}\n',
        encoding="utf-8",
    )
    assert set(read_done(path, "medical", tasks)) == {"1"}


def test_even_canary_selection():
    tasks = [{"id": index} for index in range(1, 11)]
    assert [item["id"] for item in select_evenly(tasks, 5)] == [1, 3, 5, 8, 10]
