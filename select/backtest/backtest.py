import pandas as pd
from .metrics import calculate_metrics


def backtest(prices: pd.Series, signals: pd.Series, fee_rate=0.0005):
    prices = prices.astype(float)
    ret = prices.pct_change().fillna(0.0)
    pos = signals.shift(1).fillna(0.0)
    strat_ret = ret * pos
    # transaction fee on signal change
    change = signals.diff().abs().fillna(0.0)
    fee = change * fee_rate
    strat_ret = strat_ret - fee
    equity = (1 + strat_ret).cumprod()
    metrics = calculate_metrics(strat_ret)
    return equity, metrics, strat_ret
