from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# 首版使用 Yahoo Finance 作為研究資料來源；正式環境應加入 TWSE/TAIFEX 官方資料校驗。
TICKERS = [
    "0050.TW","2330.TW","2454.TW","2317.TW","2308.TW","3711.TW",
    "2383.TW","3037.TW","2303.TW","2382.TW","2881.TW",
    "TSM","NVDA","AMD","MU","ASML","^SOX","^IXIC","^GSPC","^DJI",
    "^VIX","^TNX","^FVX","DX-Y.NYB","GC=F","CL=F","TWD=X","^TWII","^N225","^HSI","000001.SS","EWT"
]

def load_market_data(years=5):
    end = datetime.utcnow().date() + timedelta(days=1)
    start = end - timedelta(days=365 * years + 30)
    result = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                             auto_adjust=False, progress=False, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                result[ticker] = df[["Open","High","Low","Close","Volume"]].copy()
        except Exception:
            continue
    if "0050.TW" not in result:
        raise RuntimeError("無法取得 0050.TW 歷史資料。請稍後重試或更換資料供應商。")
    return result
