
# stock_prediction_system.py

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 获取股票数据
def load_data(symbol="000001", start="20220101", end="20240501"):
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
    df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
    df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['return'] = df['close'].pct_change()
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

# 构建时间序列样本
def create_sequences(X, y, window=10):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 回测策略
def backtest(returns, signals, title="Strategy"):
    strategy_returns = returns * signals.reshape(-1)
    cumulative = (1 + strategy_returns).cumprod()
    market = (1 + returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label=title)
    plt.plot(market, label='Market')
    plt.legend()
    plt.title('Backtest Result')
    plt.show()

def main():
    df = load_data()
    features = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'return']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df['label'].values

    # LSTM 输入
    X_seq, y_seq = create_sequences(X_scaled, y)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # 训练 LSTM
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    preds = model.predict(X_test) > 0.5
    acc = accuracy_score(y_test, preds)
    print(f"LSTM Accuracy: {acc:.2f}")

    returns = df.iloc[-len(preds):]['return'].values
    backtest(returns, preds, title="LSTM Strategy")

    # 训练 XGBoost
    X_flat = X_scaled
    y_flat = df['label'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_flat, y_flat, test_size=0.2, shuffle=False)
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_tr, y_tr)
    y_pred = model_xgb.predict(X_te)
    acc_xgb = accuracy_score(y_te, y_pred)
    print(f"XGBoost Accuracy: {acc_xgb:.2f}")

if __name__ == "__main__":
    main()
