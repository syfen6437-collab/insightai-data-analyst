# 数问 InsightAI

数问 InsightAI 是一个纯软件的数据分析 Web 应用。使用者可以用中文提出问题，系统识别相关数据和字段，生成只读 SQL，在本地执行，并返回结果表、指标、图表和分析报告。

项目支持医疗、通信、金融等结构化数据，也支持从 Markdown 或 HTML 表格中提取证据并完成计算。程序使用 Streamlit 提供界面，DuckDB 在本地执行分析，模型接口只负责自然语言理解和查询生成。

## 目录

```text
app.py                         Streamlit 入口
insightai/                     数据导入、SQL 安全、查询和报告模块
scripts/                       官方数据准备与任务评测脚本
tests/                         核心测试
sample_data/                   小型示例数据，可直接运行
01_作品技术文档/               技术文档
02_作品演示视频/               演示视频
04_作品运行结果/               脱敏截图、公开样例和汇总指标
05_展示PPT/                    展示材料
06_辅助材料/                  使用说明与数据合规说明
```

## 快速启动

Windows 10/11、Python 3.10 至 3.13：

```powershell
.\start.ps1
```

启动脚本会创建本地虚拟环境并安装 `requirements.txt` 中的依赖，然后打开 Streamlit 服务。没有 API 密钥时也可以查看示例数据、运行公开 SQL 样例和体验安全校验。需要生成新的自然语言查询时，在启动前配置环境变量：

```powershell
$env:INSIGHTAI_API_KEY = Read-Host "请输入模型 API Key"
.\start.ps1
```

密钥只保存在当前终端进程中，不应写入代码、日志、截图或提交文件。

## 官方数据

为了避免公开分发原始数据，本仓库不包含官方数据包、DuckDB 缓存和完整官方任务结果。使用者如果拥有相应数据和使用权限，可以将数据包放在本地后执行：

```powershell
python scripts\prepare_official_data.py --source <本地数据包ZIP> --output official_cache
```

准备完成后，程序会自动识别官方缓存；没有缓存时仍可使用示例数据和自定义 CSV、Excel、SQLite 文件。

## 主要能力

- 自然语言生成查询，并同时保留标准 SQL 与本地执行 SQL。
- 只允许单条只读查询，拦截写入、DDL、环境操作和多语句。
- 支持 CSV、XLSX、SQLite 和本地 DuckDB 分析。
- 根据结果结构生成表格、指标卡、柱状图和趋势图。
- 对文档表格任务按任务编号提取证据，保留计算过程。
- 页面、报告和示例截图默认脱敏敏感字段。

## 评测脚本

```powershell
python scripts\run_official_evaluation.py --all --resume
python scripts\run_official_evaluation.py --dataset telecom --resume
```

评测脚本需要本地准备官方数据和模型接口。仓库中的 `04_作品运行结果` 仅保存公开样例、脱敏截图和汇总指标，不包含完整官方任务数据。

## 许可证与数据说明

代码和自行制作的材料仅供学习、研究和软件演示使用。官方数据及其题目文本不随本仓库分发，使用者应自行确认数据来源、授权范围和隐私要求。
