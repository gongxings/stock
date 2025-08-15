import pandas as pd, io


def to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode('utf-8-sig')


def to_excel_bytes(dfs: dict, index: bool = False) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=index)
    return output.getvalue()
