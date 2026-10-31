"""Daily 0050 multi-horizon forecast pipeline."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_market_data
from features import build_features
from model import fit_latest_model, save_report, evaluate_horizon, compare_models_walk_forward
from news import collect_news
from market_context import update_institutional_history, add_institutional_features, latest_institutional_features, build_market_briefing, sync_official_0050_latest

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
            row["f_0050_volume_ratio20"] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 and volume.rolling(20).mean().iloc[-1] else np.nan
            ma60 = close.rolling(60).mean().iloc[-1]
            row["f_0050_ma60_gap"] = float(close.iloc[-1] / ma60 - 1) if pd.notna(ma60) and ma60 else np.nan
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            row["f_0050_macd"] = float(macd.iloc[-1] / close.iloc[-1])
            row["f_0050_macd_signal"] = float(signal.iloc[-1] / close.iloc[-1])
            row["f_0050_macd_hist"] = float((macd.iloc[-1] - signal.iloc[-1]) / close.iloc[-1])
            ma20 = close.rolling(20).mean().iloc[-1]
            std20 = close.rolling(20).std().iloc[-1]
            row["f_0050_bollinger_pctb"] = float((close.iloc[-1] - (ma20 - 2 * std20)) / (4 * std20)) if pd.notna(std20) and std20 else np.nan
            high = frame["High"].dropna().astype(float)
            low = frame["Low"].dropna().astype(float)
            previous = close.shift(1)
            tr = pd.concat([(high - low), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            row["f_0050_atr14_pct"] = float(atr / close.iloc[-1]) if pd.notna(atr) and close.iloc[-1] else np.nan
            ret20 = close.pct_change(20).iloc[-1]
            row["f_0050_roc20"] = float(ret20) if pd.notna(ret20) else np.nan
    return pd.DataFrame([{key: row.get(key, np.nan) for key in columns}])

def main():
    now = datetime.now(timezone.utc)
    raw = load_market_data(years=5)
    raw, official_close_check = sync_official_0050_latest(raw)
    data = build_features(raw)
    taiwan_prices = raw["0050.TW"].dropna(subset=["Close"])
    latest_taiwan_date = pd.Timestamp(taiwan_prices.index[-1]).date().isoformat()
    institutional_path = ROOT / "site" / "data" / "institutional_history.json"
    institutional_records, institutional_errors = update_institutional_history(
        taiwan_prices.index, institutional_path, backfill_sessions=60
    )
    data = add_institutional_features(data, institutional_records)
    latest_flow = latest_institutional_features(institutional_records, latest_taiwan_date)
    labeled = data.dropna(subset=["target_1d"]).copy()
    features = [x for x in labeled.columns if x.startswith("f_")]
    labeled = labeled.dropna(subset=features, how="all")
    if len(labeled) < 300:
        raise RuntimeError(f"Not enough labeled observations: {len(labeled)}")

    model_comparison = compare_models_walk_forward(data, target_col="target_1d", test_window=80)
    selected_model = model_comparison.get("selected_model", "Logistic Regression")
    forecasts, next_row = {}, None
    feature_names = None
    for horizon in (1, 2, 3):
        target = f"target_{horizon}d"
        horizon_model = selected_model if horizon == 1 else "Logistic Regression"
        report, predictions = evaluate_horizon(data, target, horizon=horizon, test_window=40, model_name=horizon_model)
        estimator, model_features = fit_latest_model(data, target_col=target, model_name=horizon_model)
        if next_row is None:
            feature_names = model_features
            next_row = build_next_session_row(raw, feature_names)
        for key, value in latest_flow.items():
            if key in next_row.columns:
                next_row.loc[0, key] = value
        up = float(np.clip(estimator.predict_proba(next_row[model_features])[:, 1][0], 0.01, 0.99))
        forecasts[f"{horizon}d"] = {
            "direction": "UP" if up >= .5 else "DOWN",
            "up_probability": round(up, 6),
            "down_probability": round(1-up, 6),
            "backtest": report,
            "model_name": horizon_model,
            "target_definition": f"相對最近一個台灣交易日收盤價，未來第 {horizon} 個台灣交易日收盤方向（不是開盤跳空預測）"
        }
        if horizon == 1:
            save_report(report, predictions)

    # Separate next-session opening-gap model: next open vs latest Taiwan close.
    gap_report, gap_predictions = evaluate_horizon(
        data, "target_open_up", horizon=1, test_window=40, model_name=selected_model
    )
    gap_estimator, gap_features = fit_latest_model(
        data, target_col="target_open_up", model_name=selected_model
    )
    gap_row = build_next_session_row(raw, gap_features)
    for key, value in latest_flow.items():
        if key in gap_row.columns:
            gap_row.loc[0, key] = value
    gap_up = float(np.clip(gap_estimator.predict_proba(gap_row[gap_features])[:, 1][0], 0.01, 0.99))

    # Current market stress is a transparent rule-based overlay, not a hidden
    # rewrite of the statistical forecast probability.
    stress_inputs = {"^SOX": -0.03, "^IXIC": -0.02, "TSM": -0.03}
    stress_observations = {}
    for ticker, threshold in stress_inputs.items():
        frame = raw.get(ticker)
        series = frame["Close"].dropna() if frame is not None else pd.Series(dtype=float)
        value = float(series.pct_change().dropna().iloc[-1]) if len(series) > 1 else None
        stress_observations[ticker] = {"return": value, "threshold": threshold,
                                       "triggered": bool(value is not None and value <= threshold)}
    stress_count = sum(1 for x in stress_observations.values() if x["triggered"])
    if stress_count == 3:
        risk_level, risk_label = "HIGH", "高風險：SOX、NASDAQ、台積電 ADR 同時重挫"
    elif stress_count == 2:
        risk_level, risk_label = "ELEVATED", "風險升高：三項國際指標中有兩項重挫"
    elif stress_count == 1:
        risk_level, risk_label = "WATCH", "風險觀察：一項國際指標達重挫門檻"
    else:
        risk_level, risk_label = "NORMAL", "未觸發國際市場同步重挫條件"

    # Compare live directional votes across all three calibrated model families.
    live_model_probabilities = {}
    for candidate in ("Logistic Regression", "Random Forest", "Extra Trees"):
        candidate_model, candidate_features = fit_latest_model(data, target_col="target_1d", model_name=candidate)
        candidate_row = build_next_session_row(raw, candidate_features)
        for key, value in latest_flow.items():
            if key in candidate_row.columns:
                candidate_row.loc[0, key] = value
        live_model_probabilities[candidate] = float(np.clip(
            candidate_model.predict_proba(candidate_row[candidate_features])[:, 1][0], 0.01, 0.99
        ))
    selected_report = next((x for x in model_comparison.get("models", [])
                            if x.get("name") == selected_model), forecasts["1d"]["backtest"])
    live_votes = [p >= 0.5 for p in live_model_probabilities.values()]
    signal_consensus = max(sum(live_votes), len(live_votes) - sum(live_votes)) / max(len(live_votes), 1)
    data_quality = float(next_row[feature_names].notna().mean()) if feature_names else 0.0
    confidence = {
        "data_quality_score": round(data_quality, 4),
        "available_features": int(next_row[feature_names].notna().sum()) if feature_names else 0,
        "total_features": len(feature_names or []),
        "signal_consensus": round(signal_consensus, 4),
        "model_probabilities": live_model_probabilities,
        "historical_balanced_accuracy": selected_report.get("balanced_accuracy"),
        "historical_brier_score": selected_report.get("brier_score"),
        "historical_test_count": selected_report.get("n_test"),
        "label": "可信度需搭配資料品質、模型一致性及樣本外回測解讀；非獲利保證"
    }
    risk_assessment = {
        "level": risk_level, "label": risk_label, "trigger_count": stress_count,
        "observations": stress_observations,
        "rule": "SOX <= -3%, NASDAQ <= -2%, TSM ADR <= -3%; 兩項觸發為風險升高，三項同時觸發為高風險"
    }

    prices = raw["0050.TW"].dropna(subset=["Close"])
    latest_date = pd.Timestamp(prices.index[-1]).date().isoformat()
    latest_close = float(prices["Close"].iloc[-1])
    if latest_date != official_close_check["date"]:
        raise RuntimeError(f"內部日期檢查不一致：行情序列={latest_date}、TWSE={official_close_check['date']}")
    latest_close = float(official_close_check["close"])
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
    market_briefing = build_market_briefing(
        raw, institutional_records, news, latest_taiwan_date, institutional_errors
    )
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
        "price_validation":official_close_check,
        "forecasts":forecasts,
        "opening_gap_forecast":{
            "direction":"UP" if gap_up >= 0.5 else "DOWN",
            "up_probability":round(gap_up,6),"down_probability":round(1-gap_up,6),
            "backtest":gap_report,"model_name":selected_model,
            "target_definition":"下一台灣交易日開盤價相對最近一個台灣交易日收盤價的漲跌方向"
        },
        "market_risk":risk_assessment,
        "confidence":confidence,
        "model_comparison":model_comparison,
        "history":history,
        "forecast":{"direction":forecasts["1d"]["direction"],"up_probability":p1,
                    "down_probability":forecasts["1d"]["down_probability"],
                    "target_session":"下一個台灣交易日（未接官方交易日曆）"},
        "backtest":{"method":"purged expanding-window walk-forward",
                    "accuracy":report1["accuracy"],"up_recall":report1["up_recall"],
                    "coverage":1.0,"base_rate":report1["base_rate"],"n_test":report1["n_test"]},
        "explanation":{"model_interpretation":interpretation,"market_factors":factors,
                       "news":news,"market_briefing":market_briefing,
                       "summary":"交叉閱讀量化模型、前一個台灣交易日法人籌碼、前一個已完成美股交易時段及國內外新聞；新聞事件只作解釋參考，不直接改寫模型機率。市場恐慌規則獨立顯示，不隱藏改寫模型機率。"},
        "model":{"name":selected_model,"probability_calibration":"sigmoid / chronological time-series folds","selection_method":model_comparison.get("method"),"selection_rule":model_comparison.get("selection_rule"),"probability_bounds":[0.01,0.99],"feature_count":len(feature_names),"training_rows":len(labeled)},
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
            "美股主要市場特徵已按台灣開盤前可得的最近已完成美股交易時段對齊；其他市場時區及資料供應商延遲仍需持續監控。",
            "新聞僅為 RSS 標題關鍵字整理，尚未納入模型訓練或歷史回測；法人說明會及財報事件須有可回溯的歷史公告資料，才適合納入訓練。",
            "TWSE法人資料初期回填最近60個交易日，較長期歷史尚未完整；法人特徵僅使用已公開的前一交易日資料。",
            "美股市場摘要依紐約時間最後完成的常規交易時段判定；遇休市或資料供應商延遲時，會以實際資料日期顯示，不保證一定是前一個日曆日。",
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
