import numpy as np
import pandas as pd


def calculate_metrics(return_series: pd.Series, rf_rate=0.0, periods_per_year=252):
    ret = return_series.dropna()
    if len(ret) == 0:
        return {'Annualized Return': '0.00%', 'Cumulative Return': '0.00%', 'Annualized Volatility': '0.00%',
                'Max Drawdown': '0.00%', 'Sharpe Ratio': '0.00', 'Win Rate': '0.00%', 'Trades': 0}
    cum = (1 + ret).cumprod()
    cum_return = cum.iloc[-1] - 1.0
    days = len(ret)
    ann_return = (1 + cum_return) ** (periods_per_year / days) - 1 if days > 0 else 0.0
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    peak = cum.cummax()
    dd = (cum / peak - 1.0)
    max_dd = dd.min()
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0.0
    win_rate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0.0
    trades = int((ret.ne(0)).sum())
    return {'Annualized Return': f"{ann_return:.2%}", 'Cumulative Return': f"{cum_return:.2%}",
            'Annualized Volatility': f"{ann_vol:.2%}", 'Max Drawdown': f"{max_dd:.2%}", 'Sharpe Ratio': f"{sharpe:.2f}",
            'Win Rate': f"{win_rate:.2%}", 'Trades': trades}
