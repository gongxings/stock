import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class DoubleMA(BaseStrategy):
    name = "DoubleMA (SMA Cross)"

    def __init__(self, short_window=5, long_window=20):
        self.short = int(short_window);
        self.long = int(long_window)

    def generate_signals(self, data):
        df = data.copy()
        df['ma_s'] = df['close'].rolling(self.short).mean()
        df['ma_l'] = df['close'].rolling(self.long).mean()
        sig = (df['ma_s'] > df['ma_l']).astype(int)
        sig.iloc[:max(self.short, self.long)] = 0
        return sig.fillna(0)


class EMACross(BaseStrategy):
    name = "EMACross (EMA Cross)"

    def __init__(self, short_window=12, long_window=26):
        self.short = int(short_window);
        self.long = int(long_window)

    def generate_signals(self, data):
        df = data.copy()
        df['ema_s'] = df['close'].ewm(span=self.short, adjust=False).mean()
        df['ema_l'] = df['close'].ewm(span=self.long, adjust=False).mean()
        sig = (df['ema_s'] > df['ema_l']).astype(int)
        sig.iloc[:max(self.short, self.long)] = 0
        return sig.fillna(0)


class RSI_MeanReversion(BaseStrategy):
    name = "RSI Mean Reversion"

    def __init__(self, period=14, buy_th=30, sell_th=70):
        self.period = int(period);
        self.buy_th = float(buy_th);
        self.sell_th = float(sell_th)

    def _rsi(self, close):
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(self.period).mean()
        roll_down = down.rolling(self.period).mean()
        rs = roll_up / (roll_down.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, data):
        df = data.copy()
        rsi = self._rsi(df['close'])
        sig = pd.Series(0, index=df.index)
        sig[rsi < self.buy_th] = 1
        sig[rsi > self.sell_th] = 0
        sig = sig.replace(to_replace=0, method='ffill').fillna(0)
        return sig


class MACD_Signal(BaseStrategy):
    name = "MACD Signal"

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = int(fast);
        self.slow = int(slow);
        self.signal = int(signal)

    def generate_signals(self, data):
        df = data.copy()
        ema_f = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_s = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_f - ema_s
        sigline = macd.ewm(span=self.signal, adjust=False).mean()
        sig = (macd > sigline).astype(int)
        sig.iloc[:self.slow + self.signal] = 0
        return sig.fillna(0)


class Boll_Breakout(BaseStrategy):
    name = "Bollinger Breakout"

    def __init__(self, window=20, n_std=2.0):
        self.window = int(window);
        self.n = float(n_std)

    def generate_signals(self, data):
        df = data.copy()
        mid = df['close'].rolling(self.window).mean()
        std = df['close'].rolling(self.window).std(ddof=0)
        up = mid + self.n * std;
        low = mid - self.n * std
        sig = (df['close'] > up).astype(int)
        sig.iloc[:self.window] = 0
        return sig.fillna(0)


class Boll_MeanRevert(BaseStrategy):
    name = "Bollinger Mean Reversion"

    def __init__(self, window=20, n_std=2.0):
        self.window = int(window);
        self.n = float(n_std)

    def generate_signals(self, data):
        df = data.copy()
        mid = df['close'].rolling(self.window).mean()
        low = mid - self.n * df['close'].rolling(self.window).std(ddof=0)
        sig = (df['close'] < low).astype(int)
        sig[df['close'] >= mid] = 0
        sig.iloc[:self.window] = 0
        return sig.fillna(0)


class Donchian_Breakout(BaseStrategy):
    name = "Donchian Breakout"

    def __init__(self, nh=20, nl=10):
        self.nh = int(nh);
        self.nl = int(nl)

    def generate_signals(self, data):
        df = data.copy()
        high = df['high'].rolling(self.nh).max().shift(1)
        sig = (df['close'] > high).astype(int)
        sig.iloc[:self.nh] = 0
        return sig.fillna(0)


class ROC_Momentum(BaseStrategy):
    name = "ROC Momentum"

    def __init__(self, window=20):
        self.window = int(window)

    def generate_signals(self, data):
        df = data.copy()
        roc = df['close'].pct_change(self.window)
        sig = (roc > 0).astype(int)
        sig.iloc[:self.window + 1] = 0
        return sig.fillna(0)


class MASlope(BaseStrategy):
    name = "MA Slope Positive"

    def __init__(self, window=20, lb=3):
        self.window = int(window);
        self.lb = int(lb)

    def generate_signals(self, data):
        df = data.copy()
        ma = df['close'].rolling(self.window).mean()
        slope = ma.diff(self.lb)
        sig = (slope > 0).astype(int)
        sig.iloc[:self.window + self.lb] = 0
        return sig.fillna(0)


class Volume_Spike(BaseStrategy):
    name = "Volume Spike + Above MA"

    def __init__(self, vw=20, mul=2.0, mw=20):
        self.vw = int(vw);
        self.mul = float(mul);
        self.mw = int(mw)

    def generate_signals(self, data):
        df = data.copy()
        vma = df['volume'].rolling(self.vw).mean()
        ma = df['close'].rolling(self.mw).mean()
        sig = ((df['volume'] > self.mul * vma) & (df['close'] > ma)).astype(int)
        sig.iloc[:max(self.vw, self.mw)] = 0
        return sig.fillna(0)


class BuyAndHold(BaseStrategy):
    name = "Buy and Hold (Benchmark)"

    def generate_signals(self, data):
        sig = pd.Series(1, index=data.index);
        sig.iloc[0] = 0
        return sig
