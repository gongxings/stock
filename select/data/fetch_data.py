import pandas as pd


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # standardize common Chinese & English column names
    rename_map = {'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume',
                  '成交额': 'amount',
                  'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    df = df.rename(columns=rename_map)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    # ensure required cols
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def get_stock_data(code: str, start_date: str, end_date: str):
    """Get daily stock data using akshare. Returns DataFrame with columns: date, open, high, low, close, volume.
    code can be '000001' or '000001.SZ' etc.
    start_date/end_date format: 'YYYYMMDD' or None."""
    try:
        import akshare as ak
    except Exception as e:
        print('akshare import failed:', e)
        return pd.DataFrame()
    symbol = code
    if '.' not in code and len(code) == 6:
        if code.startswith(('0', '3')):
            symbol = code + '.SZ'
        elif code.startswith('6'):
            symbol = code + '.SH'
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start_date, end_date=end_date, adjust='qfq')
        if df is None or df.empty:
            return pd.DataFrame()
        df = _normalize_dataframe(df)
        # filter by start/end if provided
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print('get_stock_data error:', e)
        return pd.DataFrame()
