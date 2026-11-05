"""Fetch a lightweight 5-minute market snapshot for the Taiwan trading session.

This is a market-data monitor, not a retrained or calibrated intraday forecast.
"""
from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "site" / "data" / "intraday.json"
TAIPEI = ZoneInfo("Asia/Taipei")

WATCHLIST = {
    "0050.TW": "元大台灣50",
    "2330.TW": "台積電",
    "^TWII": "台灣加權指數",
    "^SOX": "費城半導體指數",
    "^IXIC": "NASDAQ 綜合指數",
    "TSM": "台積電 ADR",
    "NVDA": "NVIDIA",
    "^VIX": "VIX 波動率指數",
    "NQ=F": "NASDAQ 100 期貨",
    "ES=F": "S&P 500 期貨",
}

def taipei_timestamp(value) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.to_pydatetime().replace(tzinfo=TAIPEI)
    return stamp.tz_convert(TAIPEI).to_pydatetime()

def fetch_quote(ticker: str, label: str, today: str) -> dict:
    result = {"ticker": ticker, "name": label, "status": "unavailable"}
    try:
        bars = yf.Ticker(ticker).history(
            period="1d", interval="5m", auto_adjust=False, prepost=True, timeout=15
        )
        if bars is None or bars.empty:
            result["message"] = "資料供應商目前沒有回傳 5 分鐘行情"
            return result

        bars = bars.dropna(subset=["Close"]).copy()
        bars["taipei_time"] = [taipei_timestamp(x) for x in bars.index]
        today_bars = bars[[x.date().isoformat() == today for x in bars["taipei_time"]]]
        if today_bars.empty:
            result["latest_bar_time"] = bars["taipei_time"].iloc[-1].isoformat(timespec="seconds")
            result["message"] = "目前沒有當日行情；保留為非即時狀態"
            return result

        latest = today_bars.iloc[-1]
        latest_time = today_bars["taipei_time"].iloc[-1]
        previous_close = None
        try:
            daily = yf.Ticker(ticker).history(
                period="5d", interval="1d", auto_adjust=False, prepost=False, timeout=15
            )
            if daily is not None and not daily.empty:
                daily = daily.dropna(subset=["Close"])
                prior = []
                for idx, row in daily.iterrows():
                    day = taipei_timestamp(idx).date().isoformat()
                    if day < today:
                        prior.append(float(row["Close"]))
                if prior:
                    previous_close = prior[-1]
        except Exception:
            previous_close = None

        closes = today_bars["Close"].astype(float)
        previous_bar_close = float(closes.iloc[-2]) if len(closes) > 1 else None
        last_price = float(latest["Close"])
        change_pct = (last_price / previous_close - 1) * 100 if previous_close else None
        five_min_pct = (last_price / previous_bar_close - 1) * 100 if previous_bar_close else None
        volume = float(today_bars["Volume"].fillna(0).sum()) if "Volume" in today_bars else None

        result.update({
            "status": "live",
            "latest_bar_time": latest_time.isoformat(timespec="seconds"),
            "price": last_price,
            "previous_close": previous_close,
            "change_pct": round(change_pct, 4) if change_pct is not None else None,
            "five_min_change_pct": round(five_min_pct, 4) if five_min_pct is not None else None,
            "open": float(today_bars["Open"].dropna().iloc[0]) if today_bars["Open"].notna().any() else None,
            "high": float(today_bars["High"].max()) if today_bars["High"].notna().any() else None,
            "low": float(today_bars["Low"].min()) if today_bars["Low"].notna().any() else None,
            "volume": int(volume) if volume is not None else None,
            "bars": [
                {
                    "time": row["taipei_time"].isoformat(timespec="minutes"),
                    "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                    "high": float(row["High"]) if pd.notna(row["High"]) else None,
                    "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                    "close": float(row["Close"]) if pd.notna(row["Close"]) else None,
                    "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                }
                for _, row in today_bars.tail(78).iterrows()
            ],
        })
    except Exception as exc:
        result["message"] = f"{type(exc).__name__}: {str(exc)[:180]}"
    return result

def main():
    now = datetime.now(TAIPEI)
    today = now.date().isoformat()
    quotes = [fetch_quote(ticker, label, today) for ticker, label in WATCHLIST.items()]
    etf = next((q for q in quotes if q["ticker"] == "0050.TW"), None)
    in_cash_hours = time(9, 30) <= now.time().replace(tzinfo=None) <= time(13, 35)
    etf_is_today = bool(etf and etf.get("status") == "live"
                        and str(etf.get("latest_bar_time", "")).startswith(today))
    if etf_is_today and in_cash_hours:
        status = "live"
        label = "盤中行情更新中"
    elif etf_is_today:
        status = "latest_today"
        label = "已取得今日行情；目前非盤中更新時段"
    else:
        status = "no_live_data"
        label = "尚未取得今日 0050 盤中行情"
    payload = {
        "status": status,
        "label": label,
        "trading_date_taipei": today,
        "generated_at_taipei": now.isoformat(timespec="seconds"),
        "refresh_interval_minutes": 5,
        "market_hours_taipei": "09:30-13:30（台灣證券交易日）",
        "source": "Yahoo Finance 5-minute bars; indicative data, not an official exchange feed",
        "quotes": quotes,
        "note": "盤中行情為參考資料，可能有供應商延遲或缺漏；不會冒充 TWSE 官方收盤價，也不會直接改寫每日量化模型機率。收盤後由每日模型流程以官方收盤資料重新計算下一交易日預測。",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": status, "trading_date_taipei": today,
        "generated_at_taipei": payload["generated_at_taipei"],
        "quotes_live": sum(q.get("status") == "live" for q in quotes),
        "output": str(OUTPUT.relative_to(ROOT)),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
