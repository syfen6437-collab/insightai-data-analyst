from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insightai.official_data import EXPECTED_TASKS, prepare_official_data


def main() -> int:
    parser = argparse.ArgumentParser(description="准备数问 InsightAI 官方赛题 1 数据缓存")
    parser.add_argument("--source", required=True, help="主办方外层 ZIP 或赛题 1 内层 ZIP")
    parser.add_argument("--output", default=str(ROOT / "official_cache"))
    args = parser.parse_args()
    manifest = prepare_official_data(args.source, args.output, print)
    actual = {key: manifest["structured"].get(key, manifest["unstructured"].get(key))["tasks"] for key in EXPECTED_TASKS}
    if actual != EXPECTED_TASKS or manifest["total_tasks"] != 1467:
        raise SystemExit(f"任务数量校验失败：{actual}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
