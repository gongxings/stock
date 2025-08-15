import streamlit as st
import pandas as pd
from data.fetch_data import get_stock_data
from data.sentiment import attach_popularity
from data.boards import get_board_list, get_board_members
from strategy.implementations import *
from strategy.sentiment_strategies import *
from backtest.backtest import backtest
from backtest.portfolio import backtest_portfolio, parse_weights_input, periodic_rebalance_momentum
from backtest.metrics import calculate_metrics
from optimize.grid_search import grid_search
from reporting.exporter import to_csv_bytes, to_excel_bytes

st.set_page_config(page_title='股票研究系统 Pro+ 完整版', layout='wide')
st.title('股票研究系统 Pro+ 完整版')

# Sidebar global settings
with st.sidebar:
    st.header('全局设置')
    stock_code = st.text_input('股票代码（示例：000001.SZ）', value='000001.SZ')
    start_date = st.text_input('开始日期 (YYYYMMDD)', value='20220101')
    end_date = st.text_input('结束日期 (YYYYMMDD)', value='20241231')
    pop_source = st.selectbox('热门度来源', ['Auto (体量代理)', 'Volume Buzz (体量代理)', 'Upload CSV（自定义）'])
    uploaded = None
    if pop_source.startswith('Upload'):
        uploaded = st.file_uploader('上传 CSV，包含 date,pop,sentiment 列', type=['csv'])


@st.cache_data
def load_data_attach(code, sd, ed, pop_source_key, uploaded_file):
    df = get_stock_data(code, sd, ed)
    if df is None or df.empty:
        return pd.DataFrame()
    upload_df = None
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception:
            upload_df = None
    source_key = 'upload' if upload_df is not None else ('volume_buzz' if 'Volume' in pop_source_key else 'auto')
    df = attach_popularity(df, source=('upload' if upload_df is not None else source_key), upload_df=upload_df,
                           vol_window=20)
    return df


tabs = st.tabs(['单票回测', '组合回测', '板块轮动', '参数优化', '导出报表'])

# -------- 单票回测 --------
with tabs[0]:
    st.header('单票回测')
    df = load_data_attach(stock_code, start_date, end_date, pop_source, uploaded)
    if df is None or df.empty:
        st.warning('未获取到数据，请检查代码和日期范围')
    else:
        st.dataframe(df.tail(5))
        # strategy selection
        strat_list = [DoubleMA, EMACross, RSI_MeanReversion, MACD_Signal, Boll_Breakout, Boll_MeanRevert,
                      Donchian_Breakout, ROC_Momentum, MASlope, Volume_Spike, BuyAndHold, PopFilter_DoubleMA,
                      PopMomentum_MACD, Buzz_RSI, Sentiment_EMA, HotRank_Breakout]
        strat_names = [cls.name for cls in strat_list]
        sel = st.selectbox('选择策略', strat_names)
        cls = {c.name: c for c in strat_list}[sel]
        params = {}
        if 'DoubleMA' in sel:
            params['short_window'] = st.number_input('短期窗口', 2, 120, 5)
            params['long_window'] = st.number_input('长期窗口', 3, 250, 20)
        if st.button('运行单票回测'):
            try:
                strat = cls(**params) if params else cls()
                sig = strat.generate_signals(df)
                equity, metrics, ret = backtest(df['close'], sig, fee_rate=0.0005)
                st.line_chart(pd.DataFrame({'date': df['date'], 'equity': equity}).set_index('date'))
                st.json(metrics)
                st.session_state['last_single_equity'] = pd.Series(equity.values, index=df['date'])
                st.session_state['last_single_metrics'] = metrics
            except Exception as e:
                st.error(f'回测失败: {e}')

# -------- 组合回测 --------
with tabs[1]:
    st.header('多票组合回测')
    codes_str = st.text_area('股票代码列表（逗号/空格分隔）', value='000001.SZ, 600000.SH')
    weights_text = st.text_input('可选：自定义权重（示例：600519.SH=0.3,000001.SZ=0.7），若留空则等权', value='')
    fee_rate_p = st.number_input('手续费率（组合）', 0.0, 0.01, 0.0005, 0.0001)
    if st.button('运行组合回测'):
        raw = [c.strip() for c in codes_str.replace('\n', ',').replace(' ', ',').split(',') if c.strip()]
        frames = {}
        for c in raw:
            dfi = load_data_attach(c, start_date, end_date, pop_source, uploaded)
            if dfi is None or dfi.empty:
                st.warning(f'{c} 获取失败或数据为空')
            else:
                frames[c] = dfi[['date', 'close']]
        if not frames:
            st.warning('无有效股票数据')
        else:
            w_dict, err = parse_weights_input(weights_text, list(frames.keys()))
            if err:
                st.error(f'权重输入错误：{err}')
            else:
                try:
                    equity, port_ret = backtest_portfolio(frames, weights=w_dict, fee_rate=fee_rate_p)
                    st.line_chart(
                        pd.DataFrame({'date': list(frames.values())[0]['date'], 'equity': equity}).set_index('date'))
                    m = calculate_metrics(port_ret)
                    st.json(m)
                    st.session_state['last_port_equity'] = equity
                    st.session_state['last_port_metrics'] = m
                except Exception as e:
                    st.error(f'组合回测失败: {e}')

# -------- 板块轮动 --------
with tabs[2]:
    st.header('板块轮动')
    kind = st.selectbox('板块类型', ['industry', 'concept'])
    lookback = st.number_input('动量回看天数', 20, 250, 60, 5)
    top_k = st.number_input('每期持有板块数', 1, 10, 3, 1)
    freq = st.selectbox('调仓频率', ['M (月度)', 'W (周度)', 'Q (季度)'], index=0)
    impact_cost = st.number_input('冲击成本（按换手率扣除，比例）', 0.0, 0.01, 0.001, 0.0001)
    max_turn_penalty = st.number_input('换手惩罚系数（换手>1时放大成本）', 0.0, 10.0, 1.0, 0.1)
    if st.button('获取板块列表/刷新'):
        blist = get_board_list(kind)
        if blist.empty:
            st.warning('未获取到板块列表（接口或网络问题）')
        else:
            st.session_state['board_list'] = blist
    blist = st.session_state.get('board_list')
    if blist is not None and not blist.empty:
        st.dataframe(blist.head(50))
    codes = st.text_input('参与轮动的板块代码（留空则使用上表前20）', value='')
    if st.button('运行轮动回测'):
        use_codes = [c.strip() for c in codes.split(',') if c.strip()] if codes.strip() else (
            blist['board_code'].head(20).tolist() if blist is not None else [])
        price_frames = {}
        for bcode in use_codes:
            cons = get_board_members(bcode, kind=kind).head(10)
            members = {}
            for _, row in cons.iterrows():
                code = str(row['code'])
                dfi = load_data_attach(code, start_date, end_date, pop_source, uploaded)
                if dfi is not None and not dfi.empty:
                    members[code] = dfi[['date', 'close']]
            if not members:
                continue
            wide = pd.concat([s.set_index('date')['close'].rename(k) for k, s in members.items()],
                             axis=1).ffill().dropna(how='all')
            eq = (1 + wide.pct_change().fillna(0.0).mean(axis=1)).cumprod()
            price_frames[bcode] = pd.DataFrame({'date': eq.index, 'close': eq.values})
        if not price_frames:
            st.warning('未成功构造任何板块价格序列')
        else:
            freq_key = {'M (月度)': 'M', 'W (周度)': 'W', 'Q (季度)': 'Q'}[freq]
            equity, pos_df = periodic_rebalance_momentum(price_frames, lookback=int(lookback), top_k=int(top_k),
                                                         freq=freq_key, impact_cost=float(impact_cost),
                                                         max_turnover_penalty=float(max_turn_penalty))
            st.line_chart(
                pd.DataFrame({'date': list(price_frames.values())[0]['date'], 'equity': equity}).set_index('date'))
            if not pos_df.empty:
                st.dataframe(pos_df[['date', 'hold', 'turnover', 'impact_cost']].head(200))

# -------- 参数优化 --------
with tabs[3]:
    st.header('参数网格优化（简单）')
    opt_strategy = st.selectbox('优化策略', [cls.name for cls in
                                             [DoubleMA, EMACross, RSI_MeanReversion, MACD_Signal, Boll_Breakout,
                                              Boll_MeanRevert, Donchian_Breakout, ROC_Momentum, MASlope, Volume_Spike,
                                              PopFilter_DoubleMA, PopMomentum_MACD, Buzz_RSI, Sentiment_EMA,
                                              HotRank_Breakout]])
    space_text = st.text_area('参数空间 (示例: short_window=5,10; long_window=20,50)',
                              'short_window=5,10,20; long_window=20,30,60')
    metric_key = st.selectbox('排序指标', ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Cumulative Return'])
    if st.button('开始网格搜索'):
        params = {}
        try:
            parts = [p.strip() for p in space_text.split(';') if p.strip()]
            for p in parts:
                k, vs = p.split('=')
                vals = [int(x) if x.isdigit() else float(x) for x in vs.split(',')]
                params[k.strip()] = vals
        except Exception as e:
            st.error(f'参数解析失败: {e}')
            params = {}
        if params:
            df_base = load_data_attach(stock_code, start_date, end_date, pop_source, uploaded)


            def _build(p):
                cls_map = {cls.name: cls for cls in
                           [DoubleMA, EMACross, RSI_MeanReversion, MACD_Signal, Boll_Breakout, Boll_MeanRevert,
                            Donchian_Breakout, ROC_Momentum, MASlope, Volume_Spike, PopFilter_DoubleMA,
                            PopMomentum_MACD, Buzz_RSI, Sentiment_EMA, HotRank_Breakout]}
                c = cls_map[opt_strategy]
                try:
                    return c(**p)
                except Exception:
                    return c()


            def _run(strat):
                sig = strat.generate_signals(df_base)
                equity, metrics, ret = backtest(df_base['close'], sig, fee_rate=0.0005)
                return metrics


            res = grid_search(params, _build, _run)
            if not res.empty:
                st.dataframe(
                    res.sort_values(metric_key, ascending=(metric_key in ['Max Drawdown'])).reset_index(drop=True))
                st.session_state['opt_result'] = res

# -------- 导出报表 --------
with tabs[4]:
    st.header('导出报表')
    if 'last_single_equity' in st.session_state and 'last_single_metrics' in st.session_state:
        df_eq = pd.DataFrame({'date': st.session_state['last_single_equity'].index,
                              'equity': st.session_state['last_single_equity'].values})
        st.download_button('下载 单票资金曲线 CSV', data=to_csv_bytes(df_eq, index=False),
                           file_name='single_equity.csv', mime='text/csv')
        st.download_button('下载 单票结果 Excel', data=to_excel_bytes(
            {'equity': df_eq, 'metrics': pd.DataFrame([st.session_state['last_single_metrics']])}),
                           file_name='single_result.xlsx')
    else:
        st.info('尚无单票回测缓存结果')

    if 'last_port_equity' in st.session_state and 'last_port_metrics' in st.session_state:
        df_eq2 = pd.DataFrame(
            {'date': st.session_state['last_port_equity'].index, 'equity': st.session_state['last_port_equity'].values})
        st.download_button('下载 组合资金曲线 CSV', data=to_csv_bytes(df_eq2, index=False),
                           file_name='portfolio_equity.csv')
        st.download_button('下载 组合结果 Excel', data=to_excel_bytes(
            {'equity': df_eq2, 'metrics': pd.DataFrame([st.session_state['last_port_metrics']])}),
                           file_name='portfolio_result.xlsx')
    else:
        st.info('尚无组合回测缓存结果')
