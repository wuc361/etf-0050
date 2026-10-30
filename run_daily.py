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
    history_frame = raw["0050.TW"].dropna(subset=["Open", "High", "Low", "Close"]).tail(120)
    history = [
        {"date": pd.Timestamp(i).date().isoformat(), "open": float(x["Open"]),
         "high": float(x["High"]), "low": float(x["Low"]), "close": float(x["Close"]),
         "volume": float(x["Volume"]) if pd.notna(x["Volume"]) else 0}
        for i, x in history_frame.iterrows()
    ]
    news = collect_news()
    factors = []
    for ticker, label in [("^SOX","費城半導體指數"),("^IXIC","NASDAQ"),("^GSPC","S&P 500"),("TSM","台積電 ADR"),("2330.TW","台積電")]:
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
