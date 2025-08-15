#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Date: 2025/8/5 12:18
Desc: 使用 AkShare 获取 东方财富-ETF 行情
"""

import akshare as ak
import pandas as pd


def fund_etf_spot_em() -> pd.DataFrame:
    """
    东方财富-ETF 实时行情（使用 AkShare）
    :return: ETF 实时行情
    :rtype: pandas.DataFrame
    """
    try:
        df = ak.fund_etf_spot_em()
        return df
    except Exception as e:
        print(f"获取实时行情失败: {e}")
        return pd.DataFrame()


def fund_etf_hist_em(
        symbol: str = "513500",
        period: str = "daily",
        start_date: str = "20000101",
        end_date: str = "20500101",
        adjust: str = "",
) -> pd.DataFrame:
    """
    东方财富-ETF 历史行情（日周月K线），使用 AkShare
    :param symbol: ETF 代码
    :param period: 'daily', 'weekly', 'monthly'
    :param start_date: 开始日期，格式如 '20200101'
    :param end_date: 结束日期，格式如 '20230101'
    :param adjust: 复权方式：'', 'qfq'（前复权）, 'hfq'（后复权）
    :return: 历史行情数据
    """
    period_map = {"daily": "day", "weekly": "week", "monthly": "month"}
    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period=period_map[period],
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        return df
    except Exception as e:
        print(f"获取历史日线数据失败: {e}")
        return pd.DataFrame()


def fund_etf_hist_min_em(
        symbol: str = "513500",
        period: str = "5",
        adjust: str = "",
        start_date: str = "1979-09-01 09:32:00",
        end_date: str = "2222-01-01 09:32:00",
) -> pd.DataFrame:
    """
    东方财富-ETF 分钟线行情，使用 AkShare
    :param symbol: ETF 代码
    :param period: 分钟周期：'1', '5', '15', '30', '60'
    :param adjust: 复权：'' 或 'qfq'（前复权），注意：分钟线通常不支持复权
    :param start_date: 起始时间
    :param end_date: 截止时间
    :return: 分钟线数据
    """
    try:
        df = ak.fund_etf_hist_min_em(symbol=symbol, period=period, adjust=adjust)
        # AkShare 返回的时间是字符串，转换为 datetime 并支持切片
        df["时间"] = pd.to_datetime(df["时间"])
        df.set_index("时间", inplace=True)
        df = df[start_date:end_date]
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"获取分钟线数据失败: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # 1. 获取实时行情
    spot_df = fund_etf_spot_em()
    print("=== ETF 实时行情 ===")
    print(spot_df.head())

    # 2. 获取历史日线（后复权）
    hist_hfq_df = fund_etf_hist_em(
        symbol="513500",
        period="daily",
        start_date="20200101",
        end_date="20230101",
        adjust="hfq",
    )
    print("\n=== ETF 历史日线（后复权）===")
    print(hist_hfq_df.head())

    # 3. 获取历史分钟线（5分钟）
    min_df = fund_etf_hist_min_em(
        symbol="513500",
        period="5",
        adjust="",
        start_date="2023-01-01 09:30:00",
        end_date="2023-01-03 15:00:00",
    )
    print("\n=== ETF 5分钟线 ===")
    print(min_df.head())
