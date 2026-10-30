import numpy as np
import pandas as pd

FACTOR_REGISTRY = [
 {"id":i+1,"factor":name,"group":group,"v1_status":status,"note":note}
 for i,(name,group,status,note) in enumerate([
 ("NASDAQ綜合指數","國際市場","可計算","隔夜報酬"),
 ("S&P 500","國際市場","可計算","隔夜報酬"),
 ("費城半導體SOX","國際市場","可計算","隔夜報酬"),
 ("道瓊指數","國際市場","可計算","隔夜報酬"),
 ("NASDAQ 100期貨","國際市場","待接API","需可回溯期貨資料"),
 ("SOX期貨","國際市場","待接API","需可回溯期貨資料"),
 ("台積電ADR","國際市場","可計算","隔夜報酬"),
 ("NVIDIA","國際市場","可計算","隔夜報酬"),
 ("AMD","國際市場","可計算","隔夜報酬"),
 ("美股大型科技股廣度","國際市場","待建置","需成分股歷史資料"),
 ("台積電2330","核心成分股","可計算","前一日報酬及量價"),
 ("聯發科2454","核心成分股","可計算","前一日報酬及量價"),
 ("鴻海2317","核心成分股","可計算","前一日報酬及量價"),
 ("台達電2308","核心成分股","可計算","前一日報酬及量價"),
 ("日月光3711","核心成分股","可計算","前一日報酬及量價"),
 ("台光電2383","核心成分股","可計算","前一日報酬及量價"),
 ("欣興3037","核心成分股","可計算","前一日報酬及量價"),
 ("聯電2303","核心成分股","可計算","前一日報酬及量價"),
 ("廣達2382","核心成分股","可計算","前一日報酬及量價"),
 ("富邦金2881","核心成分股","可計算","前一日報酬及量價"),
 ("NVIDIA波動率","產業","可計算","20日波動率"),
 ("美光MU","產業","待接資料","補入MU日行情"),
 ("SK海力士","產業","待接資料","需韓股資料"),
 ("ASML","產業","可計算","隔夜報酬"),
 ("ADR與台股價差","產業","待建置","需匯率及ADR換算"),
 ("AI伺服器族群","產業","待建置","動態成分股指數"),
 ("PCB族群","產業","待建置","動態成分股指數"),
 ("散熱族群","產業","待建置","動態成分股指數"),
 ("記憶體族群","產業","待建置","動態成分股指數"),
 ("半導體設備族群","產業","待建置","動態成分股指數"),
 ("美元兌新台幣","總經","可計算","匯率報酬"),
 ("美元指數DXY","總經","可計算","隔夜報酬"),
 ("美國10年債殖利率","總經","待接API","需FRED或其他資料源"),
 ("美國2年債殖利率","總經","待接API","需FRED或其他資料源"),
 ("VIX","總經","可計算","變化率"),
 ("黃金","總經","可計算","期貨報酬"),
 ("原油","總經","可計算","期貨報酬"),
 ("中國及香港股市","總經","可計算","隔夜報酬"),
 ("外資現貨買賣超","籌碼","已接TWSE BFI82U","全上市市場金額；以前一交易日資料預測"),
 ("投信買賣超","籌碼","已接TWSE BFI82U","全上市市場金額；以前一交易日資料預測"),
 ("自營商買賣超","籌碼","已接TWSE BFI82U","自行買賣及避險合計；以前一交易日資料預測"),
 ("0050個別ETF三大法人買賣超","籌碼","已接TWSE T86","外資、投信、自營商及合計；單位股數"),
 ("0050法人近5日累計","籌碼","已接TWSE T86","累計買賣超股數；目前歷史回填最近60個交易日"),
 ("外資台指期未平倉","籌碼","待接TAIFEX","使用前一交易日已公開資料"),
 ("台指期正逆價差","籌碼","待接TAIFEX","盤前可用前一日結算資料"),
 ("融資融券變化","籌碼","待接TWSE","使用前一交易日資料"),
 ("台股成交量與市場廣度","籌碼","待接TWSE","使用前一交易日資料"),
 ("台積電重大新聞情緒","新聞","待接新聞API","需時間戳及去重"),
 ("AI產業新聞情緒","新聞","待接新聞API","需時間戳及去重"),
 ("Fed及利率新聞","新聞","待接新聞API","需時間戳及去重"),
 ("0050 RSI/MACD/均線","技術","可計算","前一交易日可得"),
 ("0050前日報酬/跳空/量能","技術","可計算","前一交易日可得"),
 ])
]

def _ret(s):
    return s.pct_change()

def build_features(raw):
    target = raw["0050.TW"]["Close"].rename("close").to_frame()
    # Target day D is predicted using Taiwan data through D-1 only.
    # Horizon labels: on prediction row D, features are available through D-1.
    # 1D compares close D vs D-1; 2D compares D+1 vs D-1; 3D compares D+2 vs D-1.
    for horizon in (1, 2, 3):
        future_close = target["close"].shift(-(horizon - 1))
        previous_close = target["close"].shift(1)
        target[f"target_{horizon}d"] = (future_close > previous_close).astype(float)
        target.loc[future_close.isna() | previous_close.isna(), f"target_{horizon}d"] = np.nan
    target["target"] = target["target_1d"]
    X = pd.DataFrame(index=target.index)
    for ticker, df in raw.items():
        close = df["Close"].reindex(X.index).ffill()
        vol = df["Volume"].reindex(X.index).ffill()
        # US-market observations on calendar date D-1 are shifted to Taiwan session D.
        # All inputs are lagged at least one Taiwan session: Taiwan stocks from D-1,
        # overseas closes from the latest completed overseas session before Taiwan open.
        shift_n = 1
        r = _ret(close).shift(shift_n)
        safe = ticker.replace("^","").replace(".","_").replace("=","_").replace("-","_")
        X[f"f_{safe}_ret1"] = r
        X[f"f_{safe}_ret5"] = close.pct_change(5).shift(shift_n)
        if ticker == "0050.TW":
            X["f_0050_rsi14"] = _rsi(close).shift(1)
            X["f_0050_ma5_gap"] = (close / close.rolling(5).mean() - 1).shift(1)
            X["f_0050_ma20_gap"] = (close / close.rolling(20).mean() - 1).shift(1)
            X["f_0050_volatility20"] = close.pct_change().rolling(20).std().shift(1)
            X["f_0050_volume_change"] = vol.pct_change().shift(1)
            X["f_0050_volume_ratio20"] = (vol / vol.rolling(20).mean()).shift(1)
            X["f_0050_ma60_gap"] = (close / close.rolling(60).mean() - 1).shift(1)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            X["f_0050_macd"] = (macd / close).shift(1)
            X["f_0050_macd_signal"] = (macd_signal / close).shift(1)
            X["f_0050_macd_hist"] = ((macd - macd_signal) / close).shift(1)
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            X["f_0050_bollinger_pctb"] = ((close - (ma20 - 2 * std20)) / (4 * std20)).shift(1)
            high = df["High"].reindex(X.index).ffill()
            low = df["Low"].reindex(X.index).ffill()
            previous_close = close.shift(1)
            true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
            X["f_0050_atr14_pct"] = (true_range.rolling(14).mean() / close).shift(1)
            X["f_0050_roc20"] = close.pct_change(20).shift(1)
    # All feature values are shifted at least one Taiwan session to prevent using target-day data.
    X = X.replace([np.inf,-np.inf], np.nan)
    out = pd.concat([target, X], axis=1)
    return out

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
