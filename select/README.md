# 股票研究系统 Pro+ （完整工程）

包含：
- 使用 AkShare 获取 A 股日线数据（data/fetch_data.py）
- 技术策略与舆情策略（strategy/下）共计15+策略
- 单票回测、组合回测（自定义权重校验）、板块轮动（周/月/季/冲击成本/换手惩罚）
- 网格参数优化（optimize/grid_search.py）
- 报表导出 CSV/Excel（reporting/exporter.py）
- 前端：Streamlit（app.py）

运行：
1. 创建虚拟环境并安装依赖：
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
2. 启动应用：
   streamlit run app.py

说明：AkShare 需要联网获取数据。若网络受限，可在 data/fetch_data.py 中替换为本地数据加载。

生成日期: 2025-08-15T02:41:48.489279
