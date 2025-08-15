import pandas as pd


def get_board_list(kind: str = 'industry') -> pd.DataFrame:
    try:
        import akshare as ak
        if kind == 'industry':
            df = ak.stock_board_industry_name_em()
        else:
            df = ak.stock_board_concept_name_em()
        # simple normalization
        cols = df.columns.tolist()
        code_col = cols[0];
        name_col = cols[1]
        return pd.DataFrame({'board_code': df[code_col], 'board_name': df[name_col]})
    except Exception as e:
        print('get_board_list error:', e)
        return pd.DataFrame(columns=['board_code', 'board_name'])


def get_board_members(board_code: str, kind: str = 'industry') -> pd.DataFrame:
    try:
        import akshare as ak
        if kind == 'industry':
            df = ak.stock_board_industry_cons_em(symbol=board_code)
        else:
            df = ak.stock_board_concept_cons_em(symbol=board_code)
        rename_map = {'代码': 'code', '名称': 'name', '股票代码': 'code', '股票名称': 'name', 'symbol': 'code'}
        df = df.rename(columns=rename_map)
        if 'code' in df.columns and 'name' in df.columns:
            return df[['code', 'name']]
        return df
    except Exception as e:
        print('get_board_members error:', e)
        return pd.DataFrame(columns=['code', 'name'])
