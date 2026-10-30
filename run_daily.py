"""Daily 0050 multi-horizon forecast pipeline."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_market_data
from features import build_features
from model import fit_latest_model, save_report, evaluate_horizon
from news import collect_news

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "site" / "data" / "latest.json"

def build_next_session_row(raw, columns):
    row = {}
    for ticker, frame in raw.items():
        close = frame["Close"].dropna().astype(float)
        if close.empty:
            continue
        safe = ticker.replace("^", "").replace(".", "_").replace("=", "_").replace("-", "_")
        r1, r5 = close.pct_change().dropna(), close.pct_change(5).dropna()
        row[f"f_{safe}_ret1"] = float(r1.iloc[-1]) if len(r1) else np.nan
        row[f"f_{safe}_ret5"] = float(r5.iloc[-1]) if len(r5) else np.nan
        if ticker == "0050.TW":
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
            rsi = 100 - 100 / (1 + gain / loss)
            row["f_0050_rsi14"] = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
            ma5, ma20 = close.rolling(5).mean().iloc[-1], close.rolling(20).mean().iloc[-1]
            row["f_0050_ma5_gap"] = float(close.iloc[-1] / ma5 - 1) if pd.notna(ma5) and ma5 else np.nan
            row["f_0050_ma20_gap"] = float(close.iloc[-1] / ma20 - 1) if pd.notna(ma20) and ma20 else np.nan
            v20 = close.pct_change().rolling(20).std().iloc[-1]
            row["f_0050_volatility20"] = float(v20) if pd.notna(v20) else np.nan
            volume = frame["Volume"].dropna().astype(float)
            vc = volume.pct_change().dropna()
            row["f_0050_volume_change"] = float(vc.iloc[-1]) if len(vc) else np.nan
    return pd.DataFrame([{key: row.get(key, np.nan) for key in columns}])

def main():
    now = datetime.now(timezone.utc)
    raw = load_market_data(years=5)
    data = build_features(raw)
    labeled = data.dropna(subset=["target_1d"]).copy()
    features = [x for x in labeled.columns if x.startswith("f_")]
    labeled = labeled.dropna(subset=features, how="all")
    if len(labeled) < 300:
        raise RuntimeError(f"Not enough labeled observations: {len(labeled)}")

    forecasts, next_row = {}, None
    feature_names = None
    for horizon in (1, 2, 3):
        target = f"target_{horizon}d"
        report, predictions = evaluate_horizon(data, target, horizon=horizon, test_window=40)
        estimator, model_features = fit_latest_model(data, target_col=target)
        if next_row is None:
            feature_names = model_features
            next_row = build_next_session_row(raw, feature_names)
        up = float(estimator.predict_proba(next_row[model_features])[:, 1][0])
        forecasts[f"{horizon}d"] = {
            "direction": "UP" if up >= .5 else "DOWN",
            "up_probability": round(up, 6),
            "down_probability": round(1-up, 6),
            "backtest": report,
            "target_definition": f"相對最新收盤價，未來第 {horizon} 個台灣交易日收盤方向"
        }
        if horizon == 1:
            save_report(report, predictions)

    prices = raw["0050.TW"].dropna(subset=["Close"])
    latest_date = pd.Timestamp(prices.index[-1]).date().isoformat()
    latest_close = float(prices["Close"].iloc[-1])
    history_frame = raw["0050.TW"].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    close_series = history_frame["Close"].astype(float)
    ma5 = close_series.rolling(5).mean()
    ma20 = close_series.rolling(20).mean()
    ma60 = close_series.rolling(60).mean()
    delta = close_series.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi14 = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    bb_mid = close_series.rolling(20).mean()
    bb_std = close_series.rolling(20).std()
    bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    history_frame = history_frame.tail(120)
    history = []
    for idx, row in history_frame.iterrows():
        history.append({
            "date": pd.Timestamp(idx).date().isoformat(),
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
            "volume": float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0,
            "ma5": float(ma5.loc[idx]) if pd.notna(ma5.loc[idx]) else None,
            "ma20": float(ma20.loc[idx]) if pd.notna(ma20.loc[idx]) else None,
            "ma60": float(ma60.loc[idx]) if pd.notna(ma60.loc[idx]) else None,
            "rsi14": float(rsi14.loc[idx]) if pd.notna(rsi14.loc[idx]) else None,
            "macd": float(macd_line.loc[idx]) if pd.notna(macd_line.loc[idx]) else None,
            "macd_signal": float(macd_signal.loc[idx]) if pd.notna(macd_signal.loc[idx]) else None,
            "macd_hist": float(macd_hist.loc[idx]) if pd.notna(macd_hist.loc[idx]) else None,
            "bb_upper": float(bb_upper.loc[idx]) if pd.notna(bb_upper.loc[idx]) else None,
            "bb_mid": float(bb_mid.loc[idx]) if pd.notna(bb_mid.loc[idx]) else None,
            "bb_lower": float(bb_lower.loc[idx]) if pd.notna(bb_lower.loc[idx]) else None
        })
    news = collect_news()
    factors = []
    for ticker, label in [
        ("^SOX","費城半導體指數"),("^IXIC","NASDAQ"),("^GSPC","S&P 500"),
        ("^DJI","道瓊指數"),("TSM","台積電 ADR"),("NVDA","NVIDIA"),
        ("2330.TW","台積電"),("^TWII","台灣加權指數"),("EWT","台灣 ETF"),
        ("TWD=X","美元兌新台幣"),("^VIX","VIX 波動率指數"),
        ("^TNX","美國10年期公債殖利率"),("^FVX","美國5年期公債殖利率")
    ]:
        frame = raw.get(ticker)
        if frame is not None and len(frame["Close"].dropna()) > 1:
            series = frame["Close"].dropna()
            change = float(series.pct_change().dropna().iloc[-1])
            factors.append({"name":label,"latest_return":round(change,6),"tone":"偏多" if change>0 else "偏空" if change<0 else "中性"})
    p1 = forecasts["1d"]["up_probability"]
    interpretation = "量化模型偏向上漲。" if p1 >= .55 else "量化模型偏向下跌。" if p1 <= .45 else "量化模型訊號接近中性。"
    report1 = forecasts["1d"]["backtest"]
    result = {
        "project":"0050 AI Forecast",
        "generated_at_utc":now.isoformat(timespec="seconds"),
        "data_as_of":latest_date,
        "latest_close":latest_close,
        "forecasts":forecasts,
        "history":history,
        "forecast":{"direction":forecasts["1d"]["direction"],"up_probability":p1,
                    "down_probability":forecasts["1d"]["down_probability"],
                    "target_session":"下一個台灣交易日（未接官方交易日曆）"},
        "backtest":{"method":"purged expanding-window walk-forward",
                    "accuracy":report1["accuracy"],"up_recall":report1["up_recall"],
                    "coverage":1.0,"base_rate":report1["base_rate"],"n_test":report1["n_test"]},
        "explanation":{"model_interpretation":interpretation,"market_factors":factors,
                       "news":news,"summary":"交叉閱讀量化模型、國際市場、台灣市場及產業新聞；新聞目前不參與模型機率計算。"},
        "model":{"name":"Logistic Regression","feature_count":len(feature_names),"training_rows":len(labeled)},
        "data_source":"Yahoo Finance (unofficial; verify independently)",
        "reference_links":[
            {"name":"TWSE 臺灣證券交易所","url":"https://www.twse.com.tw/","purpose":"上市股票、三大法人與市場統計官方來源"},
            {"name":"TAIFEX 臺灣期貨交易所","url":"https://www.taifex.com.tw/","purpose":"台指期、法人未平倉與期貨市場資料"},
            {"name":"FRED 聯準會經濟資料庫","url":"https://fred.stlouisfed.org/","purpose":"美國利率、殖利率與總體經濟時間序列"},
            {"name":"CBOE VIX","url":"https://www.cboe.com/tradable_products/vix/","purpose":"市場隱含波動率參考"},
            {"name":"0050 元大投信","url":"https://www.yuantaetfs.com/product/detail/0050/Basic_information","purpose":"ETF 官方基本資料、成分與基金資訊"},
            {"name":"Yahoo Finance 0050","url":"https://finance.yahoo.com/quote/0050.TW/","purpose":"目前模型採用的非官方歷史行情資料"}
        ],
        "status":"ok",
        "limitations":[
            "研究用途，不構成投資建議。",
            "歷史樣本外準確率不保證未來表現。",
            "尚未接入官方台灣交易日曆；目標交易日標籤為近似說明。",
            "跨市場時區與資料發布時間仍需進一步驗證。",
            "新聞僅為 RSS 標題關鍵字整理，尚未納入模型訓練或歷史回測。",
            "2日及3日預測為獨立模型的累積方向分類，不是逐日價格路徑；期限越長，不確定性通常越高。",
            "多日回測已排除可能重疊的訓練標籤，但仍需持續檢驗模型穩定性。",
            "回測方向準確率不等同扣除交易成本後的投資報酬。"
        ]
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"ok","data_as_of":latest_date,"forecasts":forecasts,
                      "history_rows":len(history),"output":str(OUTPUT.relative_to(ROOT))},ensure_ascii=False))

if __name__ == "__main__":
    main()
