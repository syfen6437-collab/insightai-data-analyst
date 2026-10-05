from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryPlan:
    intent: str
    sql: str
    chart_type: str
    title: str
    explanation: str


def _contains(text: str, *keywords: str) -> bool:
    return all(keyword in text for keyword in keywords)


def plan_sample_query(question: str) -> QueryPlan | None:
    """Deterministic demo planner for the bundled retail dataset."""
    q = re.sub(r"\s+", "", question.lower())

    if any(word in q for word in ("删除", "删掉", "drop", "update", "修改数据库", "清空")):
        return QueryPlan("unsafe_request", "DROP TABLE orders", "table", "安全拦截", "该问题包含写入或删除意图，系统拒绝执行。")

    if _contains(q, "销售额", "总") or _contains(q, "总销售额"):
        return QueryPlan("kpi", "SELECT ROUND(SUM(sales_amount), 2) AS total_sales FROM orders", "metric", "累计销售额", "汇总订单表中的销售额。")
    if "订单数" in q or "多少订单" in q:
        return QueryPlan("kpi", "SELECT COUNT(*) AS order_count FROM orders", "metric", "订单数量", "统计订单表中的记录数。")
    if _contains(q, "平均", "订单") or "客单价" in q:
        return QueryPlan("kpi", "SELECT ROUND(AVG(sales_amount), 2) AS avg_order_value FROM orders", "metric", "平均订单金额", "计算每笔订单的平均销售额。")
    if "毛利率" in q and "品类" not in q and "区域" not in q:
        return QueryPlan("kpi", "SELECT ROUND(SUM(profit) * 100.0 / SUM(sales_amount), 2) AS gross_margin_pct FROM orders", "metric", "整体毛利率", "用总毛利除以总销售额。")
    if ("月" in q or "趋势" in q) and ("销售" in q or "营收" in q):
        return QueryPlan("trend_analysis", "SELECT substr(order_date, 1, 7) AS month, ROUND(SUM(sales_amount), 2) AS sales FROM orders GROUP BY month ORDER BY month", "line", "月度销售趋势", "按月份汇总销售额并按时间排序。")
    if ("月" in q or "趋势" in q) and "利润" in q:
        return QueryPlan("trend_analysis", "SELECT substr(order_date, 1, 7) AS month, ROUND(SUM(profit), 2) AS profit FROM orders GROUP BY month ORDER BY month", "line", "月度利润趋势", "按月份汇总利润。")
    if "季度" in q and "销售" in q:
        return QueryPlan("trend_analysis", "SELECT 'Q' || ((CAST(substr(order_date, 6, 2) AS INTEGER) - 1) / 3 + 1) AS quarter, ROUND(SUM(sales_amount), 2) AS sales FROM orders GROUP BY quarter ORDER BY quarter", "bar", "季度销售额", "将月份映射到季度后汇总。")
    if "区域" in q and "销售" in q:
        limit = " LIMIT 3" if "前三" in q or "top3" in q else ""
        return QueryPlan("ranking", f"SELECT region, ROUND(SUM(sales_amount), 2) AS sales FROM orders GROUP BY region ORDER BY sales DESC{limit}", "bar", "区域销售额", "按区域汇总销售额并降序排列。")
    if "区域" in q and "利润" in q:
        return QueryPlan("ranking", "SELECT region, ROUND(SUM(profit), 2) AS profit FROM orders GROUP BY region ORDER BY profit DESC", "bar", "区域利润", "按区域汇总利润。")
    if ("品类" in q or "类别" in q) and "销售" in q:
        return QueryPlan("ranking", "SELECT p.category, ROUND(SUM(o.sales_amount), 2) AS sales FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.category ORDER BY sales DESC", "bar", "品类销售额", "关联订单与产品表，按品类汇总。")
    if ("品类" in q or "类别" in q) and "利润" in q:
        return QueryPlan("ranking", "SELECT p.category, ROUND(SUM(o.profit), 2) AS profit FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.category ORDER BY profit DESC", "bar", "品类利润", "关联订单与产品表，按品类汇总利润。")
    if "产品" in q and ("前五" in q or "top5" in q or "最高" in q) and "销售" in q:
        return QueryPlan("ranking", "SELECT p.product_name, ROUND(SUM(o.sales_amount), 2) AS sales FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.product_name ORDER BY sales DESC LIMIT 5", "bar", "销售额最高的产品", "关联产品名称并选取销售额前五名。")
    if "客户" in q and ("前五" in q or "top5" in q or "最高" in q) and "销售" in q:
        return QueryPlan("ranking", "SELECT c.customer_name, ROUND(SUM(o.sales_amount), 2) AS sales FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.customer_name ORDER BY sales DESC LIMIT 5", "bar", "销售贡献最高的客户", "关联客户信息并选取销售额前五名。")
    if "渠道" in q and "销售" in q:
        return QueryPlan("comparison", "SELECT channel, ROUND(SUM(sales_amount), 2) AS sales FROM orders GROUP BY channel ORDER BY sales DESC", "pie", "渠道销售构成", "按渠道汇总销售额。")
    if "折扣" in q and "利润" in q:
        return QueryPlan("comparison", "SELECT CASE WHEN discount_rate < 0.1 THEN '低折扣' WHEN discount_rate < 0.2 THEN '中折扣' ELSE '高折扣' END AS discount_band, ROUND(AVG(profit), 2) AS avg_profit FROM orders GROUP BY discount_band ORDER BY avg_profit DESC", "bar", "折扣与平均利润", "将折扣率分段并比较平均利润。")
    if "退货" in q and ("区域" in q or "哪里" in q):
        return QueryPlan("comparison", "SELECT region, ROUND(AVG(returned) * 100, 2) AS return_rate_pct FROM orders GROUP BY region ORDER BY return_rate_pct DESC", "bar", "区域退货率", "按区域计算退货订单占比。")
    if "退货率" in q:
        return QueryPlan("kpi", "SELECT ROUND(AVG(returned) * 100, 2) AS return_rate_pct FROM orders", "metric", "整体退货率", "计算退货订单占全部订单的比例。")
    if "客户类型" in q or "客户分层" in q:
        return QueryPlan("comparison", "SELECT c.segment, ROUND(SUM(o.sales_amount), 2) AS sales, ROUND(SUM(o.profit), 2) AS profit FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.segment ORDER BY sales DESC", "bar", "客户类型表现", "关联客户分层，比较销售额与利润。")
    if "城市" in q and "销售" in q:
        return QueryPlan("ranking", "SELECT c.city, ROUND(SUM(o.sales_amount), 2) AS sales FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.city ORDER BY sales DESC LIMIT 10", "bar", "城市销售额前十", "关联客户城市并按销售额排名。")
    if "供应商" in q and "销售" in q:
        return QueryPlan("ranking", "SELECT p.supplier, ROUND(SUM(o.sales_amount), 2) AS sales FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.supplier ORDER BY sales DESC", "bar", "供应商相关销售额", "关联产品供应商并汇总销售额。")
    if "库存" in q and ("低" in q or "不足" in q):
        return QueryPlan("filter", "SELECT product_name, stock_qty FROM products WHERE stock_qty < 80 ORDER BY stock_qty ASC", "bar", "低库存产品", "筛选库存量低于 80 的产品。")
    if "负利润" in q or "亏损订单" in q:
        return QueryPlan("filter", "SELECT order_id, order_date, region, sales_amount, profit FROM orders WHERE profit < 0 ORDER BY profit ASC LIMIT 20", "table", "负利润订单", "筛选利润小于零的订单。")
    if "2025" in q and "销售" in q:
        return QueryPlan("filter", "SELECT ROUND(SUM(sales_amount), 2) AS sales_2025 FROM orders WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01'", "metric", "2025 年销售额", "按日期范围筛选 2025 年订单。")
    if "华东" in q and "销售" in q:
        return QueryPlan("filter", "SELECT ROUND(SUM(sales_amount), 2) AS east_sales FROM orders WHERE region = '华东'", "metric", "华东销售额", "筛选华东区域后汇总销售额。")
    if "线上" in q and "线下" in q:
        return QueryPlan("comparison", "SELECT channel, ROUND(SUM(sales_amount), 2) AS sales, ROUND(SUM(profit), 2) AS profit FROM orders WHERE channel IN ('线上', '线下') GROUP BY channel", "bar", "线上与线下表现", "比较线上和线下渠道的销售额与利润。")
    if "相关" in q and "销售" in q and "利润" in q:
        return QueryPlan("correlation", "SELECT sales_amount, profit FROM orders", "scatter", "销售额与利润关系", "返回订单级销售额与利润用于散点分析。")
    if "每天" in q or "日均" in q:
        return QueryPlan("kpi", "SELECT ROUND(AVG(daily_sales), 2) AS avg_daily_sales FROM (SELECT order_date, SUM(sales_amount) AS daily_sales FROM orders GROUP BY order_date)", "metric", "日均销售额", "先按日期汇总，再计算日均销售额。")
    if "没有" in q or "天气" in q or "员工年龄" in q:
        return None
    return None
