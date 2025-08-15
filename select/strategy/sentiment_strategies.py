import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up/(roll_down.replace(0, np.nan))
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

class PopFilter_DoubleMA(BaseStrategy):
    name = "PopFilter DoubleMA"
    def __init__(self, short_window=5, long_window=20, pop_window=60, pop_quantile=0.7):
        self.short=int(short_window); self.long=int(long_window); self.pw=int(pop_window); self.q=float(pop_quantile)
    def generate_signals(self, data):
        df = data.copy()
        df['ma_s'] = df['close'].rolling(self.short).mean()
        df['ma_l'] = df['close'].rolling(self.long).mean()
        tech = (df['ma_s'] > df['ma_l']).astype(int)
        pop_roll = df['pop'].rolling(self.pw).apply(lambda x: np.nanquantile(x, self.q), raw=True)
        pop_ok = df['pop'] >= pop_roll
        sig = (tech & pop_ok).astype(int)
        sig.iloc[:max(self.short,self.long,self.pw)] = 0
        return sig.fillna(0)

class PopMomentum_MACD(BaseStrategy):
    name = "PopMomentum + MACD"
    def __init__(self, pop_lb=5, fast=12, slow=26, signal=9):
        self.plb=int(pop_lb); self.fast=int(fast); self.slow=int(slow); self.signal=int(signal)
    def generate_signals(self, data):
        df = data.copy()
        ema_f = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_s = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_f - ema_s
        sigline = macd.ewm(span=self.signal, adjust=False).mean()
        macd_ok = macd > sigline
        pop_mom = df['pop'] - df['pop'].shift(self.plb)
        pop_ok = pop_mom > 0
        sig = (macd_ok & pop_ok).astype(int)
        sig.iloc[:max(self.slow+self.signal,self.plb)+1] = 0
        return sig.fillna(0)

class Buzz_RSI(BaseStrategy):
    name = "Buzz + RSI Trend Zone"
    def __init__(self, rsi_period=14, rsi_lower=45, rsi_upper=80, buzz_th=1.5):
        self.rp=int(rsi_period); self.rl=float(rsi_lower); self.ru=float(rsi_upper); self.bt=float(buzz_th)
    def generate_signals(self, data):
        df = data.copy()
        rsi = _rsi(df['close'], self.rp)
        buzz_ok = df['pop'] >= self.bt
        rsi_ok = (rsi >= self.rl) & (rsi <= self.ru)
        sig = (buzz_ok & rsi_ok).astype(int)
        sig.iloc[:self.rp+1] = 0
        return sig.fillna(0)

class Sentiment_EMA(BaseStrategy):
    name = "Sentiment + EMA"
    def __init__(self, short_window=10, long_window=30, sentiment_th=0.0):
        self.s=int(short_window); self.l=int(long_window); self.st=float(sentiment_th)
    def generate_signals(self, data):
        df = data.copy()
        ema_s = df['close'].ewm(span=self.s, adjust=False).mean()
        ema_l = df['close'].ewm(span=self.l, adjust=False).mean()
        tech = ema_s > ema_l
        senti_ok = df['sentiment'].fillna(-1e9) > self.st
        sig = (tech & senti_ok).astype(int)
        sig.iloc[:max(self.s,self.l)+1] = 0
        return sig.fillna(0)

class HotRank_Breakout(BaseStrategy):
    name = "HotRank Breakout"
    def __init__(self, pop_ma=20, high_window=60):
        self.pm=int(pop_ma); self.hw=int(high_window)
    def generate_signals(self, data):
        df = data.copy()
        pop_ma = df['pop'].rolling(self.pm).mean()
        pop_ok = df['pop'] > pop_ma
        rolling_high = df['high'].rolling(self.hw).max().shift(1)
        price_ok = df['close'] > rolling_high
        sig = (pop_ok & price_ok).astype(int)
        sig.iloc[:max(self.pm,self.hw)+1] = 0
        return sig.fillna(0)
