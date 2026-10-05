from __future__ import annotations

import json
import os
import re
from typing import Any
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from insightai.data import DataWorkspace
from insightai.analysis import execute_question
from insightai.llm import answer_table_with_api, plan_with_api
from insightai.official_data import STRUCTURED, UNSTRUCTURED, load_manifest, load_tasks, mask_frame
from insightai.official_engine import OfficialWorkspace, evidence_summary, load_unstructured_task, public_sample_answer, table_relevance
from insightai.offline import QueryPlan
from insightai.safety import UnsafeQueryError, validate_read_only


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "official_cache"
RESULTS = ROOT / "official_results"
SAMPLE_DB = ROOT / "sample_data" / "insightai_demo.sqlite"
PUBLIC_SQL = {
    ("medical", 1): "SELECT COUNT(*) AS 科室数量 FROM departments_wards WHERE is_active = TRUE;",
    ("medical", 2): "SELECT patient_name, contact_phone FROM patient_master_index WHERE patient_name LIKE '张%';",
    ("telecom", 1): "SELECT customer_id, customer_name, gender, json_extract_string(address_json, '$.province') AS province, json_extract_string(address_json, '$.city') AS city FROM customers WHERE json_extract_string(address_json, '$.province') = '北京市' ORDER BY customer_id",
    ("telecom", 2): "SELECT customer_id, customer_name, customer_status FROM customers WHERE customer_status = '欠费'",
    ("finance", 2): "SELECT client_type, COUNT(*) AS client_count FROM clients GROUP BY client_type",
}

st.set_page_config(page_title="数问 InsightAI", page_icon="📊", layout="wide")
st.markdown("""
<style>
:root { --ink:#17221d; --green:#177245; --mint:#edf5f1; --line:#d9e3dd; }
.stApp { background:#fbfcfb; color:var(--ink); }
[data-testid="stSidebar"] { background:#f0f5f2; border-right:1px solid var(--line); }
.block-container { padding-top:1.8rem; max-width:1320px; }
h1,h2,h3 { letter-spacing:0 !important; color:var(--ink); }
h1 { font-size:2rem !important; }
div[data-testid="stMetric"] { border-top:3px solid var(--green); padding-top:.55rem; }
.stButton button,.stDownloadButton button { border-radius:4px; border-color:var(--green); }
.stButton button[kind="primary"] { background:var(--green); color:white; }
.status { border-left:3px solid var(--green); background:var(--mint); padding:.7rem 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def manifest_data():
    return load_manifest(CACHE)


@st.cache_data(show_spinner=False)
def tasks(dataset: str):
    return load_tasks(CACHE, dataset)


def api_ready() -> bool:
    return bool(os.getenv("INSIGHTAI_API_KEY", "").strip())


def formal_results_ready() -> bool:
    path = RESULTS / "summary.json"
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        return sum(item.get("total", 0) for item in summary) == 1467 and all(item.get("missing", 0) == 0 for item in summary)
    except (OSError, ValueError, TypeError):
        return False


def evaluation_summary() -> dict[str, dict]:
    """Read the latest audit summary without making the UI depend on a fixed run."""
    path = RESULTS / "summary.json"
    if not path.exists():
        return {}
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        return {item["dataset"]: item for item in items if item.get("dataset")}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def chart_for(frame: pd.DataFrame, title: str, prefer_line: bool = False):
    if frame.empty:
        return None
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if not numeric or len(frame) > 100:
        return None
    x = next((column for column in frame.columns if column not in numeric), frame.columns[0])
    if prefer_line:
        return px.line(frame, x=x, y=numeric[:2], title=title, markers=True, color_discrete_sequence=["#177245", "#d08a1e"])
    return px.bar(frame, x=x, y=numeric[:2], title=title, color_discrete_sequence=["#177245", "#d08a1e"])


def _extract_region(question: str) -> str | None:
    """Extract a province or city name from a natural-language question."""
    text = re.sub(r"\s+", "", question or "")
    regions = (
        "北京市", "北京", "天津市", "天津", "上海市", "上海", "重庆市", "重庆",
        "河北省", "河北", "山西省", "山西", "辽宁省", "辽宁", "吉林省", "吉林",
        "黑龙江省", "黑龙江", "江苏省", "江苏", "浙江省", "浙江", "安徽省", "安徽",
        "福建省", "福建", "江西省", "江西", "山东省", "山东", "河南省", "河南",
        "湖北省", "湖北", "湖南省", "湖南", "广东省", "广东", "海南省", "海南",
        "四川省", "四川", "贵州省", "贵州", "云南省", "云南", "陕西省", "陕西",
        "甘肃省", "甘肃", "青海省", "青海", "台湾省", "台湾", "内蒙古自治区", "内蒙古",
        "广西壮族自治区", "广西", "西藏自治区", "西藏", "宁夏回族自治区", "宁夏",
        "新疆维吾尔自治区", "新疆", "广州", "深圳", "成都", "杭州",
    )
    for region in sorted(regions, key=len, reverse=True):
        if region in text:
            return region.replace("市", "").replace("省", "").replace("自治区", "")
    return None


def _extract_gender(question: str) -> str | None:
    text = re.sub(r"\s+", "", question or "")
    if "女性" in text or re.search(r"(?<!男)女(?:性)?(?:用户|客户|人员|生)?", text):
        return "女"
    if "男性" in text or re.search(r"男(?:性)?(?:用户|客户|人员|生)?", text):
        return "男"
    return None


def _is_list_intent(question: str) -> bool:
    text = re.sub(r"\s+", "", question or "")
    return any(token in text for token in ("找出", "列出", "列举", "展示", "显示", "查看", "查询明细", "所有", "哪些", "名单", "列表", "明细"))


def _is_count_intent(question: str) -> bool:
    text = re.sub(r"\s+", "", question or "")
    return any(token in text for token in ("有多少", "多少个", "数量", "人数", "总数", "共有", "统计人数"))


def _condition_pairs(question: str) -> list[tuple[str, str]]:
    text = re.sub(r"\s+", "", question or "")
    pairs: list[tuple[str, str]] = []
    region = _extract_region(text)
    if region:
        pairs.append(("省份/城市", region))
    gender = _extract_gender(text)
    if gender:
        pairs.append(("性别", gender))
    statuses = ("正常", "欠费", "停机", "在网", "离网", "活跃", "不活跃")
    for status in statuses:
        if status in text and any(token in text for token in ("状态", "用户", "客户")):
            pairs.append(("状态", status))
            break
    years = re.findall(r"(?:19|20)\d{2}", text)
    if years:
        pairs.append(("年份", "、".join(dict.fromkeys(years))))
    return pairs


def _region_json_predicate(region: str) -> str:
    """Build an exact province/city predicate for DuckDB JSON addresses."""
    safe = region.replace("'", "''")
    return (
        "(json_extract_string(address_json, '$.province') IN ('{0}', '{0}省', '{0}市') "
        "OR json_extract_string(address_json, '$.city') IN ('{0}', '{0}市') "
        "OR json_extract_string(address_json, '$.city') LIKE '{0}%')"
    ).format(safe)


def _city_from_value(value: Any) -> str | None:
    """Read city from an address JSON value as a display fallback."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict):
            city = parsed.get("city") or parsed.get("城市") or parsed.get("市")
            if city:
                return str(city).replace("市", "")
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def enhance_display_frame(frame: pd.DataFrame, question: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Add visible evidence columns without guessing over SQL-returned fields."""
    pairs = _condition_pairs(question)
    shown = frame.copy()
    rename = {"province": "省份", "city": "城市", "gender": "性别", "customer_id": "用户编号", "customer_name": "用户姓名"}
    shown.rename(columns={key: value for key, value in rename.items() if key in shown.columns}, inplace=True)
    if "城市" not in shown.columns and "address_json" in shown.columns:
        shown["城市"] = shown["address_json"].map(_city_from_value)
    if "省份" not in shown.columns and "address_json" in shown.columns:
        def province_value(value: Any) -> str | None:
            try:
                parsed = json.loads(str(value))
                if isinstance(parsed, dict):
                    return parsed.get("province") or parsed.get("省份")
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            return None
        shown["省份"] = shown["address_json"].map(province_value)
    if "address_json" in shown.columns:
        shown.drop(columns=["address_json"], inplace=True)
    if pairs and len(shown) > 0 and not (len(shown) == 1 and shown.shape[1] == 1):
        checks = "；".join(f"{label}={value}" for label, value in pairs)
        shown["查询条件核对"] = checks
    return shown, pairs


def condition_summary(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "当前问题没有识别到需要回显的筛选条件。"
    return "；".join(f"{label}：{value}" for label, value in pairs)


def deterministic_plan(dataset: str, question: str) -> QueryPlan | None:
    """Keep common demo questions deterministic and dialect-safe."""
    text = re.sub(r"\s+", "", question or "")
    if dataset == "telecom" and "用户" in text:
        region = _extract_region(text)
        gender = _extract_gender(text)
        if region and (_is_list_intent(text) or not _is_count_intent(text)):
            clauses = [_region_json_predicate(region)]
            if gender:
                clauses.append(f"gender = '{gender}'")
            where = " AND ".join(clauses)
            return QueryPlan(
                "filtered_list",
                "SELECT customer_id, customer_name, gender, "
                "json_extract_string(address_json, '$.province') AS province, "
                "json_extract_string(address_json, '$.city') AS city "
                f"FROM customers WHERE {where} ORDER BY customer_id",
                "table",
                f"{region}{'女性' if gender == '女' else '男性' if gender == '男' else ''}用户明细",
                "按地址 JSON 的省份/城市字段精确筛选" + (f"{region}" if region else "") + (f"和性别{gender}" if gender else "") + "，返回用户明细。",
            )
        if region and _is_count_intent(text):
            clauses = [_region_json_predicate(region)]
            if gender:
                clauses.append(f"gender = '{gender}'")
            where = " AND ".join(clauses)
            return QueryPlan(
                "filtered_count",
                f"SELECT COUNT(*) AS user_count FROM customers WHERE {where}",
                "metric",
                f"{region}用户数量",
                "按地址 JSON 的省份/城市字段筛选后统计用户数量。",
            )
    if dataset == "telecom" and any(word in text for word in ("每月", "月度", "趋势")) and any(word in text for word in ("账单", "收入")):
        return QueryPlan(
            "monthly_bill_trend",
            "SELECT SUBSTRING(billing_month, 1, 7) AS month, "
            "ROUND(SUM(total_amount), 2) AS total_bill_amount "
            "FROM monthly_bills WHERE billing_month IS NOT NULL "
            "GROUP BY SUBSTRING(billing_month, 1, 7) ORDER BY month ASC",
            "line",
            "每月账单总额趋势",
            "将 billing_month 截取为 YYYY-MM 后汇总 total_amount，并按月份升序排列。",
        )
    return None

if "official_history" not in st.session_state:
    st.session_state.official_history = []
if "custom_workspace" not in st.session_state:
    st.session_state.custom_workspace = DataWorkspace(SAMPLE_DB)

manifest = manifest_data()
pages = ["数据源管理", "数据概览", "智能问数", "表格问答", "分析报告", "官方评测"]
with st.sidebar:
    st.markdown("## 数问 InsightAI")
    st.caption("智能数据分析助手")
    page = st.radio("功能", pages, label_visibility="collapsed")
    st.divider()
    st.caption("官方缓存：" + ("已准备" if manifest else "未准备"))
    if api_ready():
        st.caption("DeepSeek：已配置，可现场重跑")
    elif formal_results_ready():
        st.caption("DeepSeek：正式结果已生成，现场重跑需密钥")
    else:
        st.caption("DeepSeek：未配置")
    st.caption("所有查询只读；展示数据默认脱敏。")

if page == "数据源管理":
    st.title("数据源管理")
    if manifest:
        st.markdown('<div class="status">内置数据集已完成本地缓存，可直接进入查询与评测。</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("总任务", f"{manifest['total_tasks']:,}")
        cols[1].metric("结构化任务", f"{sum(v['tasks'] for v in manifest['structured'].values()):,}")
        cols[2].metric("非结构化任务", f"{sum(v['tasks'] for v in manifest['unstructured'].values()):,}")
        cols[3].metric("结构化数据表", sum(len(v["tables"]) for v in manifest["structured"].values()))
        progress_rows = []
        for item in manifest["structured"].values():
            progress_rows.append({"任务集": item["label"], "类型": "Text2SQL", "任务数": item["tasks"], "数据表/块": len(item["tables"]), "记录数": item["rows"], "状态": "就绪"})
        for item in manifest["unstructured"].values():
            progress_rows.append({"任务集": item["label"], "类型": "表格问答", "任务数": item["tasks"], "数据表/块": item["blocks"], "记录数": None, "状态": "就绪" if item["blocks"] == item["tasks"] else "需检查"})
        st.dataframe(pd.DataFrame(progress_rows), width="stretch", hide_index=True)
    else:
        st.warning("尚未建立内置数据缓存。运行 `python scripts/prepare_official_data.py --source <数据包ZIP> --output official_cache`。")
    st.subheader("自定义数据")
    uploads = st.file_uploader("上传 CSV、Excel 或 SQLite", type=["csv", "xlsx", "xls", "sqlite", "sqlite3", "db"], accept_multiple_files=True)
    if uploads and st.button("导入自定义数据"):
        workspace = DataWorkspace()
        loaded = []
        for upload in uploads:
            loaded.extend(workspace.load_file(upload.getvalue(), upload.name))
        st.session_state.custom_workspace = workspace
        st.success("已导入：" + "、".join(loaded))

elif page == "数据概览":
    st.title("数据概览")
    if not manifest:
        st.info("官方缓存未准备，当前展示随仓库提供的脱敏示例数据；也可以在数据源管理中上传自定义文件。")
        workspace = st.session_state.custom_workspace
        sample_tables = workspace.tables()
        if sample_tables:
            table = st.selectbox("示例数据表", sample_tables)
            profile = workspace.profile(table)
            c1, c2, c3 = st.columns(3)
            c1.metric("记录数", f"{profile.rows:,}")
            c2.metric("字段数", profile.columns)
            c3.metric("空值总数", f"{sum(field['空值数'] for field in profile.fields):,}")
            st.subheader("字段画像")
            st.dataframe(pd.DataFrame(profile.fields), width="stretch", hide_index=True)
            st.subheader("示例数据")
            st.dataframe(workspace.execute(f'SELECT * FROM "{table.replace(chr(34), chr(34) + chr(34))}" LIMIT 5'), width="stretch", hide_index=True)
    else:
        labels = {key: value["label"] for key, value in STRUCTURED.items()}
        dataset = st.selectbox("业务领域", list(labels), format_func=lambda key: labels[key])
        workspace = OfficialWorkspace(CACHE, dataset)
        table = st.selectbox("数据表", workspace.tables())
        with st.spinner("正在用 SQL 聚合字段画像"):
            profile = workspace.profile(table)
        c1, c2, c3 = st.columns(3)
        c1.metric("记录数", f"{profile['rows']:,}")
        c2.metric("字段数", len(profile["fields"]))
        c3.metric("空值总数", f"{sum(field['空值数'] for field in profile['fields']):,}")
        st.subheader("字段画像")
        st.dataframe(pd.DataFrame(profile["fields"]), width="stretch", hide_index=True)
        st.subheader("脱敏样例")
        st.dataframe(profile["sample"], width="stretch", hide_index=True)
        workspace.close()

elif page == "智能问数":
    st.title("智能问数")
    if not manifest:
        st.info("当前使用随仓库提供的脱敏示例数据。准备官方数据后可切换到医疗、通信和金融任务。")
        question = st.text_area("问题", value="按区域统计销售额", height=90, key="sample_question")
        show_chart = st.checkbox("显示图表", value=True, key="sample_show_chart")
        run_sample = st.button("运行示例查询", type="primary")
        if run_sample:
            st.session_state.sample_result = execute_question(st.session_state.custom_workspace, question, prefer_api=api_ready())
        result = st.session_state.get("sample_result")
        if result is not None:
            if result.status == "成功":
                st.subheader("查询结果")
                st.caption(result.message)
                if len(result.data) == 1 and result.data.shape[1] == 1:
                    st.metric("统计结果", str(result.data.iloc[0, 0]))
                figure = chart_for(result.data, "示例查询结果", prefer_line=bool(result.plan and result.plan.chart_type == "line"))
                if show_chart and figure:
                    figure.update_layout(template="plotly_white", font_family="Microsoft YaHei")
                    st.plotly_chart(figure, width="stretch")
                st.dataframe(result.data, width="stretch", hide_index=True)
                with st.expander("示例 SQL", expanded=False):
                    st.code(result.plan.sql if result.plan else "", language="sql")
            elif result.status == "无法回答":
                st.info(result.message)
            else:
                st.error(result.message)
    else:
        labels = {key: value["label"] for key, value in STRUCTURED.items()}
        dataset = st.selectbox("领域", list(labels), index=1, format_func=lambda key: labels[key])
        official_tasks = tasks(dataset)
        task_options = {f"{item['id']}  {item['problem']}": item for item in official_tasks}
        selected_task = task_options[st.selectbox("官方任务", list(task_options))]
        question = st.text_area("问题", value=selected_task["problem"], height=90)
        public_sql = PUBLIC_SQL.get((dataset, selected_task["id"]))
        c1, c2 = st.columns(2)
        run_model = c1.button("调用 DeepSeek 生成 SQL", type="primary", disabled=not api_ready())
        run_sample = c2.button("运行公开样例 SQL", disabled=not bool(public_sql))
        show_chart = st.checkbox("显示图表", value=True, help="关闭后只展示统计结果和结果表，不绘制图表。")
        if not api_ready():
            st.caption("本机未配置密钥。公开样例仍可演示 SQL 校验、转译、执行和脱敏展示。")
        with st.expander("SQL 安全测试"):
            probe = st.text_input("待校验 SQL", value="DROP TABLE customers")
            if st.button("执行安全校验"):
                try:
                    validate_read_only(probe)
                    st.session_state.safety_message = ("success", "只读 SQL 校验通过")
                except UnsafeQueryError as exc:
                    st.session_state.safety_message = ("error", f"安全策略已拦截：{exc}")
        if "safety_message" in st.session_state:
            level, message = st.session_state.safety_message
            if level == "success":
                st.success(message)
            else:
                st.markdown(f'<div style="background:#fdecec;color:#8a1c1c;border-left:4px solid #b42318;padding:.8rem 1rem">{message}</div>', unsafe_allow_html=True)
        if run_model or run_sample:
            # A previous safety probe should not remain visible after a new query.
            st.session_state.pop("safety_message", None)
            # Each query owns its explanation. Do not let a previous question's
            # interpretation remain visible after switching dataset or task.
            st.session_state.last_plan_explanation = None
            workspace = OfficialWorkspace(CACHE, dataset)
            selected = []
            schema = ""
            repair_count = 0
            if run_model:
                fixed_plan = deterministic_plan(dataset, question)
                if fixed_plan:
                    generated_plan = fixed_plan
                    mysql_sql = fixed_plan.sql
                    st.session_state.last_plan_explanation = fixed_plan.explanation
                else:
                    selected = table_relevance(question, workspace.tables(), dictionary=workspace.dictionary)
                    schema = workspace.schema_text(selected)
                    generated_plan = plan_with_api(question, schema)
                    mysql_sql = generated_plan.sql
                    st.session_state.last_plan_explanation = generated_plan.explanation
            else:
                mysql_sql = public_sql
                st.session_state.last_plan_explanation = "运行公开样例 SQL，结果来自本地官方缓存。"
            if not mysql_sql.strip():
                workspace.close()
                st.session_state.query_notice = "系统判断当前数据缺少回答该问题所需的字段，因此没有生成 SQL，也不会编造结果。"
            else:
                result = workspace.execute_mysql(mysql_sql)
                # The model may produce valid MySQL that needs a DuckDB-specific
                # repair (for example JSON_UNQUOTE). Feed the concrete execution
                # error back once or twice instead of leaving the teacher with a
                # raw dialect failure.
                while run_model and result.status != "成功" and repair_count < 2:
                    repair_count += 1
                    generated_plan = plan_with_api(question, schema, result.error)
                    mysql_sql = generated_plan.sql
                    st.session_state.last_plan_explanation = generated_plan.explanation
                    if not mysql_sql.strip():
                        break
                    result = workspace.execute_mysql(mysql_sql)
                result.task_id, result.question = str(selected_task["id"]), question
                st.session_state.official_history.append(result)
                st.session_state.query_notice = ""
                workspace.close()
        if st.session_state.get("query_notice"):
            st.info(st.session_state.query_notice)
        if st.session_state.get("last_plan_explanation"):
            st.caption("查询理解：" + st.session_state.last_plan_explanation)
        if st.session_state.official_history:
            result = st.session_state.official_history[-1]
            if result.status == "成功":
                st.subheader("查询结果")
                shown, pairs = enhance_display_frame(result.data, result.question)
                shown = mask_frame(shown)
                st.markdown(
                    f'<div style="background:#edf5f1;border-left:4px solid #177245;padding:.8rem 1rem;margin:.4rem 0 1rem 0">'
                    f'<strong>查询条件核对</strong><br>{condition_summary(pairs)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if len(shown) == 1 and shown.shape[1] == 1:
                    value = shown.iloc[0, 0]
                    st.metric("统计结果", f"{value:,}" if isinstance(value, (int, float)) else str(value))
                c1, c2, c3 = st.columns(3)
                c1.metric("执行状态", result.status)
                c2.metric("预览行数", len(result.data))
                c3.metric("本地执行耗时", f"{result.elapsed_ms:.1f} ms")
                is_time_query = any(word in result.question for word in ("每月", "月度", "趋势", "月份"))
                figure = chart_for(shown, "查询结果", prefer_line=is_time_query)
                if show_chart and figure and not (len(shown) == 1 and shown.shape[1] == 1):
                    figure.update_layout(template="plotly_white", font_family="Microsoft YaHei")
                    st.plotly_chart(figure, width="stretch")
                st.dataframe(shown, width="stretch", hide_index=True)
                with st.expander("标准 MySQL SQL 与本地转译", expanded=False):
                    st.code(result.mysql_sql, language="sql")
                    st.caption("DuckDB 本地执行副本")
                    st.code(result.duckdb_sql, language="sql")
            else:
                st.error(result.error)

elif page == "表格问答":
    st.title("表格问答")
    if not manifest:
        st.warning("请先准备官方数据。")
    else:
        labels = {key: value["label"] for key, value in UNSTRUCTURED.items()}
        dataset = st.selectbox("任务类型", list(labels), index=1, format_func=lambda key: labels[key])
        official_tasks = tasks(dataset)
        id_options = [str(item["id"]) for item in official_tasks]
        default = id_options.index("b40962fe65f24d9") if "b40962fe65f24d9" in id_options else 0
        task_id = st.selectbox("任务 ID", id_options, index=default)
        task, block, frames = load_unstructured_task(CACHE, dataset, task_id)
        st.subheader("问题")
        table_question = st.text_area(
            "问题（可修改）",
            value=task["question"],
            height=90,
            key=f"table_question_{dataset}_{task_id}",
        )
        st.caption(f"精确索引到 {len(frames)} 张表，仅发送当前任务证据，不上传整份 Markdown。")
        for index, frame in enumerate(frames, 1):
            st.markdown(f"**来源表 {index}**")
            st.dataframe(frame.head(100), width="stretch", hide_index=True)
        if st.button("计算答案", type="primary", disabled=not api_ready()):
            with st.spinner("正在依据当前任务表格计算"):
                model_answer = answer_table_with_api(table_question, evidence_summary(frames))
                is_official_question = table_question.strip() == task["question"].strip()
                st.session_state.table_answer = (public_sample_answer(task_id, frames) if is_official_question else None) or model_answer
                st.session_state.table_answer_question = table_question
        if not api_ready():
            st.info("未配置模型密钥，当前展示证据。配置后可生成计算过程和正式答案。")
        references = {"b40962fe65f24d9": "-2.86%", "98a478f23b754a7": "2369.4"}
        if task_id in references:
            st.caption(f"公开参考答案：{references[task_id]}（仅用于样例验证）")
        if "table_answer" in st.session_state:
            answer = st.session_state.table_answer
            st.metric("系统答案", answer.get("answer", ""))
            st.write(answer.get("calculation", ""))

elif page == "分析报告":
    st.title("分析报告")
    history = st.session_state.official_history
    if not history:
        st.info("完成至少一次结构化查询后生成报告。")
    else:
        lines = ["# 数问 InsightAI 分析报告", "", "数据展示已脱敏；正式 SQL 与本地执行 SQL 分开留存。", ""]
        for index, result in enumerate(history, 1):
            lines.extend([f"## {index}. {result.question}", "", f"状态：{result.status}；耗时：{result.elapsed_ms:.1f} ms", "", "```sql", result.mysql_sql, "```", ""])
            if result.status == "成功":
                lines.extend(["结果摘要：", "```csv", mask_frame(result.data).head(20).to_csv(index=False), "```", ""])
        report = "\n".join(lines)
        st.markdown(report)
        st.download_button("下载 Markdown 报告", report, "InsightAI_官方数据分析报告.md")

elif page == "官方评测":
    st.title("官方评测")
    if not manifest:
        st.warning("请先准备官方数据。")
    else:
        rows = []
        for item in manifest["structured"].values():
            rows.append({"任务集": item["label"], "类别": "结构化", "任务数": item["tasks"], "数据状态": "就绪", "模型结果": "待运行"})
        for item in manifest["unstructured"].values():
            rows.append({"任务集": item["label"], "类别": "非结构化", "任务数": item["tasks"], "数据状态": "就绪" if item["blocks"] == item["tasks"] else f"已索引 {item['blocks']}", "模型结果": "待运行"})
        summary = evaluation_summary()
        for row, key in zip(rows, list(STRUCTURED) + list(UNSTRUCTURED)):
            if key in summary:
                item = summary[key]
                row["模型结果"] = f"成功 {item.get('success', 0)} / {item['total']}，失败 {item.get('failed', 0)}"
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        structured_total = sum(summary.get(key, {}).get("total", 0) for key in STRUCTURED)
        structured_success = sum(summary.get(key, {}).get("success", 0) for key in STRUCTURED)
        unstructured_total = sum(summary.get(key, {}).get("total", 0) for key in UNSTRUCTURED)
        unstructured_success = sum(summary.get(key, {}).get("success", 0) for key in UNSTRUCTURED)
        c1.metric("结构化 SQL 执行率", f"{structured_success / structured_total:.2%}" if structured_total else "待运行")
        c2.metric("非结构化答案非空率", f"{unstructured_success / unstructured_total:.2%}" if unstructured_total else "待运行")
        evidence = sum(summary.get(key, {}).get("evidence_retained", 0) for key in UNSTRUCTURED)
        c3.metric("证据留存率", f"{evidence / unstructured_total:.2%}" if unstructured_total else "待运行")
        c4.metric("隐藏答案准确率", "无金标准")
        st.subheader("评测口径")
        st.write("全量评测报告处理完成率、SQL 解析率、SQL 执行率、公开样例验证率，以及非结构化答案非空率和证据留存率。无公开标准答案的任务不宣称准确率。")
        st.code("python scripts/run_official_evaluation.py --all --resume", language="powershell")

