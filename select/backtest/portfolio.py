import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def align_price_frames(price_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_series = []
    for code, df in price_frames.items():
        s = df.set_index('date')['close'].astype(float).rename(code)
        all_series.append(s)
    wide = pd.concat(all_series, axis=1).sort_index()
    return wide


def equal_weight_weights(codes: List[str]) -> Dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def parse_weights_input(weights_text: str, codes: List[str]) -> Tuple[Optional[Dict[str, float]], str]:
    if not weights_text or not str(weights_text).strip():
        return None, ''
    pairs = [p.strip() for p in weights_text.replace('\n', ',').replace(' ', ',').split(',') if p.strip()]
    w = {}
    for p in pairs:
        if '=' not in p:
            return None, f"格式错误: 每项需使用 = 分隔 (示例: 600519.SH=0.3) -> 错误项 '{p}'"
        code, val = p.split('=', 1)
        code = code.strip()
        try:
            valf = float(val.strip())
        except Exception:
            return None, f"权重值必须为数字 -> 错误项 '{p}'"
        w[code] = valf
    extra = [c for c in w.keys() if c not in codes]
    if extra:
        return None, f"权重包含未在股票列表中的代码: {extra}"
    s = sum(w.values())
    if abs(s - 1.0) > 1e-4:
        return None, f"权重和必须等于1（允许微小误差±0.0001），当前和={s:.6f}"
    return w, ''


def backtest_portfolio(price_frames: Dict[str, pd.DataFrame],
                       weights: Dict[str, float] | None = None,
                       fee_rate: float = 0.0005):
    wide_prices = align_price_frames(price_frames)
    wide_prices = wide_prices.ffill().dropna(how='all')
    returns = wide_prices.pct_change().fillna(0.0)
    if weights is None:
        ws = pd.Series(equal_weight_weights(list(wide_prices.columns)))
    else:
        ws = pd.Series({c: weights.get(c, 0.0) for c in wide_prices.columns})
        total = ws.sum() if ws.sum() != 0 else 1.0
        ws = ws / total
    port_ret = (returns * ws).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    return equity, port_ret


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
    if freq == 'M':
        return index.to_series().resample('M').last().index.tolist()
    elif freq == 'W':
        return index.to_series().resample('W-FRI').last().index.tolist()
    elif freq == 'Q':
        return index.to_series().resample('Q').last().index.tolist()
    else:
        return index.to_series().resample('M').last().index.tolist()


def periodic_rebalance_momentum(price_frames: Dict[str, pd.DataFrame],
                                lookback: int = 60,
                                top_k: int = 3,
                                freq: str = 'M',
                                impact_cost: float = 0.001,
                                max_turnover_penalty: float = 1.0):
    wide = align_price_frames(price_frames)
    wide = wide.ffill().dropna(how='all')
    if wide.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    ret = wide.pct_change().fillna(0.0)
    rebalance_dates = _get_rebalance_dates(wide.index, freq)
    rebalance_dates = [d for d in rebalance_dates if d in wide.index]
    port_ret = pd.Series(0.0, index=wide.index)
    current_weights = {c: 0.0 for c in wide.columns}
    records = []
    prev_weights = current_weights.copy()
    for i, dt in enumerate(wide.index):
        if dt in rebalance_dates:
            hist_slice = wide.loc[:dt].tail(lookback + 1)
            if len(hist_slice) < 2:
                new_weights = {c: 0.0 for c in wide.columns}
            else:
                mom = hist_slice.iloc[-1] / hist_slice.iloc[0] - 1.0
                top = mom.sort_values(ascending=False).head(top_k).index.tolist()
                if top:
                    new_weights = {c: (1.0 / len(top) if c in top else 0.0) for c in wide.columns}
                else:
                    new_weights = {c: 0.0 for c in wide.columns}
            turnover = 0.0
            for c in wide.columns:
                turnover += abs(new_weights.get(c, 0.0) - prev_weights.get(c, 0.0))
            turnover = turnover / 2.0
            cost = turnover * impact_cost * (1.0 if turnover <= 1.0 else max_turnover_penalty)
            port_ret.loc[dt] = port_ret.loc[dt] - cost
            prev_weights = new_weights.copy()
            current_weights = new_weights.copy()
            records.append({'date': dt, 'weights': current_weights, 'turnover': turnover, 'impact_cost': cost,
                            'hold': [c for c, w in current_weights.items() if w > 0]})
        w = pd.Series({c: current_weights.get(c, 0.0) for c in wide.columns})
        port_ret.loc[dt] = port_ret.loc[dt] + (ret.loc[dt] * w).sum()
    equity = (1 + port_ret).cumprod()
    rec_df = pd.DataFrame(records)
    return equity, rec_df
