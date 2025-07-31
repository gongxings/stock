import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time


# 获取所有A股股票列表
def get_all_stocks():
    stock_info = ak.stock_info_a_code_name()
    return stock_info


# 获取某只股票的历史涨跌幅（动量因子）
def get_momentum(ts_code, period=60):
    try:
        df = ak.stock_zh_a_hist(symbol=ts_code, period="daily", adjust="qfq")
        if df is None or len(df) < period:
            return np.nan
        df = df.tail(period)
        if '收盘' not in df.columns:
            return np.nan
        momentum = df['收盘'].iloc[-1] / df['收盘'].iloc[0] - 1
        return momentum
    except Exception as e:
        print(f"Error fetching momentum for {ts_code}: {e}")
        return np.nan


# 获取基本面数据（PE、ROE）
def get_fundamentals(ts_code):
    try:
        df = ak.stock_financial_analysis_indicator(symbol=ts_code, start_year="2025")
        if df is None or df.empty:
            raise ValueError(f"No financial data found for stock: {ts_code}")
        pe = df.loc[0, '市盈率TTM'] if '市盈率TTM' in df.columns else np.nan
        roe = df.loc[0, '净资产收益率TTM'] if '净资产收益率TTM' in df.columns else np.nan
        return pe, roe
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return np.nan, np.nan
    except Exception as e:
        print(f"Unexpected error for stock {ts_code}: {e}")
        return np.nan, np.nan


# 单个股票因子计算
def calculate_single_stock(ts_code):
    try:
        # 动量因子（60日涨跌幅）
        momentum = get_momentum(ts_code)

        # 基本面因子（PE、ROE）
        pe, roe = get_fundamentals(ts_code)

        if pe is not None and pe > 0:
            value = 1 / pe  # 价值因子为 PE 的倒数
        else:
            value = np.nan

        # 返回单个股票的因子数据
        return {
            'ts_code': ts_code,
            'momentum': momentum,
            'value': value,
            'roe': roe
        }
    except Exception as e:
        print(f"Error processing stock {ts_code}: {e}")
        return None


# 单个股票因子计算并打分
def calculate_single_stock_with_score(ts_code):
    try:
        # 动量因子（60日涨跌幅）
        momentum = get_momentum(ts_code)

        # 基本面因子（PE、ROE）
        pe, roe = get_fundamentals(ts_code)

        if pe is not None and pe > 0:
            value = 1 / pe  # 价值因子为 PE 的倒数
        else:
            value = np.nan

        # 检查因子是否有缺失值
        if np.isnan(momentum) or np.isnan(value) or np.isnan(roe):
            return None

        # 标准化处理（Z-score）
        factors = np.array([momentum, value, roe]).reshape(1, -1)
        scaler = StandardScaler()
        factors_z = scaler.fit_transform(factors)

        # 计算综合得分
        score = np.mean(factors_z)

        # 返回单个股票的因子数据和得分
        return {
            'ts_code': ts_code,
            'momentum': momentum,
            'value': value,
            'roe': roe,
            'score': score
        }
    except Exception as e:
        print(f"Error processing stock {ts_code}: {e}")
        return None


# 主函数：生成因子并打分
def generate_factor_scores(top_n=20):
    stocks_df = get_all_stocks()
    results = []

    for index, row in stocks_df.iterrows():
        ts_code = row['code']
        name = row['name']

        print(f"Processing: {ts_code} - {name}")

        # 动量因子（60日涨跌幅）
        momentum = get_momentum(ts_code)

        # 基本面因子（PE、ROE）
        pe, roe = get_fundamentals(ts_code)

        if pe is not None and pe > 0:
            value = 1 / pe  # 价值因子为 PE 的倒数
        else:
            value = np.nan

        results.append({
            'ts_code': ts_code,
            'name': name,
            'momentum': momentum,
            'value': value,
            'roe': roe
        })

        time.sleep(0.5)  # 避免请求频率过高

    factor_df = pd.DataFrame(results)

    # 去除缺失值
    factor_df.dropna(subset=['momentum', 'value', 'roe'], inplace=True)

    # 标准化处理（Z-score）
    scaler = StandardScaler()
    factor_df[['mom_z', 'val_z', 'roe_z']] = scaler.fit_transform(factor_df[['momentum', 'value', 'roe']])

    # 等权合成综合得分
    factor_df['score'] = factor_df[['mom_z', 'val_z', 'roe_z']].mean(axis=1)

    # 排序并取前 N 名
    top_stocks = factor_df.sort_values(by='score', ascending=False).head(top_n)

    return top_stocks[['ts_code', 'name', 'momentum', 'value', 'roe', 'score']]


# 测试方法
def test():
    # 测试单个股票因子计算
    print("Testing single stock factor calculation...")
    single_stock_result = calculate_single_stock("000002")
    print(single_stock_result)

    # 测试整体因子评分
    # print("\nTesting overall factor score generation...")
    # top_stocks = generate_factor_scores(top_n=5)
    # print(top_stocks)


if __name__ == '__main__':
    # 运行测试
    # 测试单个股票打分
    result = calculate_single_stock_with_score("600004")
    print(result)
