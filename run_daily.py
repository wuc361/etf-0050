"""Daily, non-interactive pipeline for GitHub Actions (no Streamlit required)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_market_data
from features import build_features
from model import fit_latest_model, save_report, walk_forward_evaluate, evaluate_horizon
from news import collect_news

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "site" / "data" / "latest.json"


def build_next_session_row(raw: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Build features available after the latest completed observations for a next-session forecast.

    The existing training table shifts all features by one Taiwan session because row D predicts
    D's close direction using information available before D's session. For the next session,
    use the latest completed observations directly, with the same feature names/formulas.
    """
    if "0050.TW" not in raw or raw["0050.TW"].empty:
        raise RuntimeError("Missing 0050.TW price history")

    row: dict[str, float] = {}
    target_close = raw["0050.TW"]["Close"].dropna()
    if len(target_close) < 25:
        raise RuntimeError("Not enough 0050.TW history to build technical features")

    for ticker, frame in raw.items():
        close = frame["Close"].dropna().astype(float)
        if close.empty:
            continue
        safe = ticker.replace("^", "").replace(".", "_").replace("=", "_").replace("-", "_")
        ret1 = close.pct_change().dropna()
        ret5 = close.pct_change(5).dropna()
        row[f"f_{safe}_ret1"] = float(ret1.iloc[-1]) if len(ret1) else np.nan
        row[f"f_{safe}_ret5"] = float(ret5.iloc[-1]) if len(ret5) else np.nan

        if ticker == "0050.TW":
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            row["f_0050_rsi14"] = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            row["f_0050_ma5_gap"] = float(close.iloc[-1] / ma5 - 1) if pd.notna(ma5) and ma5 else np.nan
            row["f_0050_ma20_gap"] = float(close.iloc[-1] / ma20 - 1) if pd.notna(ma20) and ma20 else np.nan
            vol20 = close.pct_change().rolling(20).std().iloc[-1]
            row["f_0050_volatility20"] = float(vol20) if pd.notna(vol20) else np.nan
            volume = frame["Volume"].dropna().astype(float)
            vchange = volume.pct_change().dropna()
            row["f_0050_volume_change"] = float(vchange.iloc[-1]) if len(vchange) else np.nan

    # Match the model's feature columns exactly. Missing optional market series remain NaN
    # and are handled by the model pipeline's median imputer.
    return pd.DataFrame([{col: row.get(col, np.nan) for col in feature_columns}])


def main() -> None:
    now = datetime.now(timezone.utc)
    raw = load_market_data(years=5)
    data = build_features(raw)

    labeled = data.dropna(subset=["target"]).copy()
    features = [col for col in labeled.columns if col.startswith("f_")]
    if not features:
        raise RuntimeError("No model features were generated")
    labeled = labeled.dropna(subset=features, how="all")
    if len(labeled) < 300:
        raise RuntimeError(f"Not enough labeled observations for a meaningful run: {len(labeled)}")

    # Train and evaluate each horizon independently. Each label is a cumulative
    # direction from the latest close to the close after 1, 2, or 3 Taiwan sessions.
    horizon_results = {}
    model_features = None
    next_row = None
    for horizon in (1, 2, 3):
        target_col = f"target_{horizon}d"
        horizon_report, horizon_predictions = evaluate_horizon(data, target_col, test_window=40)
        estimator, horizon_features = fit_latest_model(data, target_col=target_col)
        if model_features is None:
            model_features = horizon_features
            next_row = build_next_session_row(raw, model_features)
        probability = float(estimator.predict_proba(next_row[horizon_features])[:, 1][0])
        horizon_results[f"{horizon}d"] = {
            "direction": "UP" if probability >= 0.5 else "DOWN",
            "up_probability": round(probability, 6),
            "down_probability": round(1 - probability, 6),
            "threshold": 0.5,
            "backtest": horizon_report,
            "target_definition": f"相對最新收盤價，未來第 {horizon} 個台灣交易日收盤方向"
        }
        if horizon == 1:
            save_report(horizon_report, horizon_predictions)
    report = horizon_results["1d"]["backtest"]
    probability_up = horizon_results["1d"]["up_probability"]
    latest_date = pd.Timestamp(raw["0050.TW"]["Close"].dropna().index[-1]).date().isoformat()
    latest_close = float(raw["0050.TW"]["Close"].dropna().iloc[-1])
    price_frame = raw["0050.TW"].dropna(subset=["Open", "High", "Low", "Close"]).tail(120).copy()
    price_frame.index = pd.to_datetime(price_frame.index)

    news_analysis = collect_news()
    market_factors = []
    for ticker, label in [("^SOX", "費城半導體指數"), ("^IXIC", "NASDAQ"), ("^GSPC", "S&P 500"), ("TSM", "台積電 ADR"), ("2330.TW", "台積電")]:
        frame = raw.get(ticker)
        if frame is not None and len(frame["Close"].dropna()) >= 2:
            series = frame["Close"].dropna()
            change = float(series.pct_change().dropna().iloc[-1])
            market_factors.append({"name": label, "latest_return": round(change, 6), "tone": "偏多" if change > 0 else "偏空" if change < 0 else "中性"})

    if probability_up >= 0.55:
        model_interpretation = "量化模型偏向上漲，但仍須觀察新聞與市場風險是否抵銷。"
    elif probability_up <= 0.45:
        model_interpretation = "量化模型偏向下跌，需留意國際市場及台灣市場是否出現支撐訊號。"
    else:
        model_interpretation = "量化模型訊號接近中性，方向優勢不明顯。"

    # Do not claim the weekday approximation is an official exchange calendar.
    result = {
        "project": "0050 AI Forecast",
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "data_as_of": latest_date,
        "latest_close": latest_close,
        "forecasts": horizon_results,
        "history": [\n            {"date": pd.Timestamp(idx).date().isoformat(), "open": float(row["Open"]),\n             "high": float(row["High"]), "low": float(row["Low"]),\n             "close": float(row["Close"]), "volume": float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0}\n            for idx, row in price_frame.iterrows()\n        ],\n        "forecast": {
            "direction": "UP" if probability_up >= 0.5 else "DOWN",
            "up_probability": round(probability_up, 6),
            "down_probability": round(1 - probability_up, 6),
            "threshold": 0.5,
            "target_session": "下一個台灣交易日（本版未接入官方交易日曆）"
        },
        "backtest": {
            "method": "expanding-window walk-forward",
            "accuracy": report["accuracy"],
            "up_recall": report["up_recall"],
            "coverage": report["coverage"],
            "base_rate": report["base_rate"],
            "n_test": report["n_test"]
        },
        "explanation": {
            "model_interpretation": model_interpretation,
            "market_factors": market_factors,
            "news": news_analysis,
            "summary": "交叉閱讀量化模型、國際市場、台灣市場及產業新聞；新聞目前不參與模型機率計算。"
        },
        "model": {
            "name": "Logistic Regression",
            "feature_count": len(model_features),
            "training_rows": int(len(labeled))
        },
        "data_source": "Yahoo Finance (unofficial; verify independently)",
        "status": "ok",
        "limitations": [
            "研究用途，不構成投資建議。",
            "歷史樣本外準確率不保證未來表現。",
            "尚未接入官方台灣交易日曆；目標交易日標籤為近似說明。",
            "跨市場時區與資料發布時間仍需進一步驗證。",
            "新聞僅為 RSS 標題關鍵字整理，尚未納入模型訓練或歷史回測。",\n            "2日及3日預測為各自訓練的累積方向分類，不是逐日價格路徑；預測期限越長，不確定性通常越高。",
            "回測方向準確率不等同扣除交易成本後的投資報酬。"
        ]
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "data_as_of": latest_date,
        "direction": result["forecast"]["direction"],
        "up_probability": result["forecast"]["up_probability"],\n        "forecasts": result["forecasts"],
        "backtest_accuracy": result["backtest"]["accuracy"],
        "output": str(OUTPUT.relative_to(ROOT))
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
