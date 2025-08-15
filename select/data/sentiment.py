import pandas as pd
import numpy as np


def build_volume_buzz_popularity(df: pd.DataFrame, vol_window: int = 20) -> pd.Series:
    vma = df['volume'].rolling(vol_window).mean()
    pop = df['volume'] / (vma.replace(0, np.nan))
    pop = pop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pop


def merge_upload_sentiment(df: pd.DataFrame, upload_df: pd.DataFrame) -> pd.DataFrame:
    udf = upload_df.copy()
    udf.columns = [c.strip().lower() for c in udf.columns]
    if 'date' not in udf.columns:
        for possible in ['日期', 'datetime', 'time', 'date_time']:
            if possible in udf.columns:
                udf['date'] = udf[possible]
                break
    if 'date' not in udf.columns:
        raise ValueError('上传文件需要包含 date 列')
    udf['date'] = pd.to_datetime(udf['date'])
    # normalize pop and sentiment columns
    if 'pop' not in udf.columns:
        if 'popularity' in udf.columns:
            udf['pop'] = udf['popularity']
        elif '热度' in udf.columns:
            udf['pop'] = udf['热度']
        else:
            udf['pop'] = pd.NA
    if 'sentiment' not in udf.columns:
        if '情绪' in udf.columns:
            udf['sentiment'] = udf['情绪']
        else:
            udf['sentiment'] = pd.NA
    merged = pd.merge(df, udf[['date', 'pop', 'sentiment']].drop_duplicates('date', keep='last'), on='date', how='left')
    return merged


def attach_popularity(df: pd.DataFrame, source: str = 'auto', upload_df: pd.DataFrame = None,
                      vol_window: int = 20) -> pd.DataFrame:
    out = df.copy()
    if source == 'upload' and upload_df is not None:
        try:
            return merge_upload_sentiment(out, upload_df)
        except Exception:
            out['pop'] = build_volume_buzz_popularity(out, vol_window)
            out['sentiment'] = pd.NA
            return out
    # auto or volume_buzz
    out['pop'] = build_volume_buzz_popularity(out, vol_window)
    out['sentiment'] = pd.NA
    return out
