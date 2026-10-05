from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insightai.official_engine import OfficialWorkspace, load_unstructured_task


def main() -> int:
    cache = ROOT / "official_cache"
    records = []
    samples = {
        "medical": json.loads((cache / "samples" / "medical.json").read_text(encoding="utf-8")),
        "telecom": json.loads((cache / "samples" / "telecom.json").read_text(encoding="utf-8")),
    }
    for dataset, tasks in samples.items():
        workspace = OfficialWorkspace(cache, dataset)
        for task in tasks:
            result = workspace.execute_mysql(task["sql"])
            records.append({"dataset": dataset, "id": task["id"], "type": "公开 SQL 执行", "expected": "SQL 可执行", "actual": result.status, "passed": result.status == "成功", "rows": len(result.data), "note": "按执行结果验证语义，不比较 SQL 字符串"})
        workspace.close()

    _, _, first = load_unstructured_task(cache, "domain_calc", "b40962fe65f24d9")
    values = [float(str(value).replace("$", "").strip()) for value in first[0].iloc[:, 1]]
    actual_first = f"{(values[0] - values[1]) / values[1] * 100:.2f}%"
    records.append({"dataset": "domain_calc", "id": "b40962fe65f24d9", "type": "公开答案计算", "expected": "-2.86%", "actual": actual_first, "passed": actual_first == "-2.86%", "rows": 2, "note": "(34-35)/35"})

    _, _, second = load_unstructured_task(cache, "domain_calc", "98a478f23b754a7")
    amounts = second[0].iloc[:, 1].astype(str).str.replace("$", "", regex=False).str.strip()
    net_2016 = float(amounts.iloc[-1])
    retail_effect = float(amounts.iloc[1])
    actual_second = f"{net_2016 - retail_effect:.1f}"
    records.append({"dataset": "domain_calc", "id": "98a478f23b754a7", "type": "公开答案计算", "expected": "2369.4", "actual": actual_second, "passed": actual_second == "2369.4", "rows": 7, "note": "2438.4-69.0"})

    output = ROOT / "official_results" / "sample_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0 if all(item["passed"] for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
