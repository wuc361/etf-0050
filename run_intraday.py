"""Fetch a lightweight 5-minute market snapshot for the Taiwan trading session.

This is a market-data monitor, not a retrained or calibrated intraday forecast.
"""
from __future__ import annotations

import json
import re
import html as html_lib
from urllib.request import Request, urlopen
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
    result = {"ticker": ticker, "name": label, "status": "unavailable", "price": None}
    try:
        bars = yf.Ticker(ticker).history(
            period="5d", interval="5m", auto_adjust=False, prepost=True, timeout=18
        )
        if bars is None or bars.empty:
            result["message"] = "資料供應商目前沒有回傳5分鐘行情"
            return result

        bars = bars.dropna(subset=["Close"]).copy()
        # Keep the exchange-local session date separate from the Taipei
        # observation date. A US session on Nov 5 spans Nov 5 and Nov 6 in Taipei.
        bars["taipei_time"] = [taipei_timestamp(x) for x in bars.index]
        bars["exchange_session_date"] = [pd.Timestamp(x).date().isoformat() for x in bars.index]
        latest_time = bars["taipei_time"].iloc[-1]
        quote_taipei_date = latest_time.date().isoformat()
        quote_exchange_date = str(bars["exchange_session_date"].iloc[-1])
        session_bars = bars[bars["exchange_session_date"] == quote_exchange_date]
        if session_bars.empty:
            result["message"] = "沒有可用的有效交易時段K線"
            return result

        latest = session_bars.iloc[-1]
        previous_close = None
        try:
            daily = yf.Ticker(ticker).history(
                period="10d", interval="1d", auto_adjust=False, prepost=False, timeout=15
            )
            if daily is not None and not daily.empty:
                daily = daily.dropna(subset=["Close"])
                prior = [
                    float(row["Close"]) for idx, row in daily.iterrows()
                    if pd.Timestamp(idx).date().isoformat() < quote_exchange_date
                ]
                if prior:
                    previous_close = prior[-1]
        except Exception:
            previous_close = None

        closes = session_bars["Close"].astype(float)
        previous_bar_close = float(closes.iloc[-2]) if len(closes) > 1 else None
        last_price = float(latest["Close"])
        change_pct = (last_price / previous_close - 1) * 100 if previous_close else None
        five_min_pct = (last_price / previous_bar_close - 1) * 100 if previous_bar_close else None
        volume = float(session_bars["Volume"].fillna(0).sum()) if "Volume" in session_bars else None
        is_today = quote_taipei_date == today
        status = "live" if is_today else "latest"
        result.update({
            "status": status,
            "market_date_taipei": quote_taipei_date,
            "exchange_session_date": quote_exchange_date,
            "is_current_taipei_date": is_today,
            "latest_bar_time": latest_time.isoformat(timespec="seconds"),
            "price": last_price,
            "previous_close": previous_close,
            "change_pct": round(change_pct, 4) if change_pct is not None else None,
            "five_min_change_pct": round(five_min_pct, 4) if five_min_pct is not None else None,
            "open": float(session_bars["Open"].dropna().iloc[0]) if session_bars["Open"].notna().any() else None,
            "high": float(session_bars["High"].max()) if session_bars["High"].notna().any() else None,
            "low": float(session_bars["Low"].min()) if session_bars["Low"].notna().any() else None,
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
                for _, row in session_bars.tail(78).iterrows()
            ],
            "message": "" if is_today else "今日尚未產生新K線；顯示最近一個有效交易日資料，並標示實際日期。",
        })
    except Exception as exc:
        result["message"] = f"{type(exc).__name__}: {str(exc)[:180]}"
    return result

def fetch_0050_margin_maintenance() -> dict:
    """Estimate 0050's margin maintenance ratio from public daily price/financing data.

    This is an aggregate position-cost estimate, not a broker's account-level ratio.
    FinMind's free per-symbol datasets provide daily close and margin transactions.
    """
    from urllib.parse import urlencode
    import os

    base_url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"User-Agent": "ETF-0050-monitor/1.0"}
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def fetch_dataset(dataset: str) -> list[dict]:
        params = {
            "dataset": dataset,
            "data_id": "0050",
            "start_date": "2024-01-01",
        }
        request = Request(
            f"{base_url}?{urlencode(params)}",
            headers=headers,
        )
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{dataset} 沒有回傳有效資料：{payload.get('msg', 'empty data')}")
        return rows

    try:
        prices = fetch_dataset("TaiwanStockPrice")
        margins = fetch_dataset("TaiwanStockMarginPurchaseShortSale")
        close_by_date = {
            str(row.get("date"))[:10]: float(row["close"])
            for row in prices
            if row.get("date") and row.get("close") not in (None, "")
        }
        margin_by_date = {
            str(row.get("date"))[:10]: row
            for row in margins
            if row.get("date")
        }

        # 0050 completed a 4-for-1 split on 2025-06-18.
        # Convert the pre-split moving cost into post-split price units.
        split_factors = {"2025-06-18": 4.0}
        moving_cost = None
        latest = None
        history = []
        for day in sorted(set(close_by_date).intersection(margin_by_date)):
            close = close_by_date[day]
            row = margin_by_date[day]
            balance = float(row.get("MarginPurchaseTodayBalance") or 0)
            buy = float(row.get("MarginPurchaseBuy") or 0)

            if day in split_factors and moving_cost is not None:
                moving_cost /= split_factors[day]

            if balance <= 0:
                moving_cost = None
                continue

            if moving_cost is None:
                # Initial observable balance: seed with that day's close.
                moving_cost = close
            else:
                # XQ-style moving average estimate. Sales and cash repayments
                # reduce the old position; the current balance less today's buys
                # is treated as the remaining old position.
                old_balance = max(balance - buy, 0.0)
                moving_cost = (
                    moving_cost * old_balance + close * buy
                ) / balance if balance else None

            if moving_cost and moving_cost > 0:
                ratio = close / (moving_cost * 0.60) * 100
                latest = {
                    "date": day,
                    "close": round(close, 4),
                    "margin_balance_lots": int(balance),
                    "margin_cost_estimate": round(moving_cost, 4),
                    "financing_ratio": 0.60,
                    "maintenance_pct": round(ratio, 2),
                }
                history.append(dict(latest))

        if not latest:
            raise ValueError("0050 融資資料與收盤價沒有可計算的共同日期")

        return {
            "status": "estimated",
            "etf_0050": latest,
            "history": history[-500:],
            "source": "FinMind 免費個股每日資料（融資交易＋收盤價）",
            "source_url": "https://finmind.github.io/tutor/TaiwanMarket/Chip/",
            "method": "依每日融資買進、融資餘額與收盤價推估移動平均融資成本；維持率＝收盤價÷（推估成本×60%）×100%。",
            "history_start": "2024-01-01",
            "note": "0050 個股市場平均估算，不是券商逐戶實際維持率；融資成數採法定最高 60%，個別帳戶與個別部位可能不同。",
            "market_wide": {
                "status": "not_calculated",
                "note": "上市／上櫃整體維持率暫不自行推算：公開資料的融資餘額張數與融資金額、擔保品市值未能在同一口徑完整核對，避免產生誤導數字。",
            },
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "etf_0050": None,
            "source": "FinMind 公開個股每日資料",
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            "market_wide": {
                "status": "not_calculated",
                "note": "上市／上櫃整體維持率暫不自行推算，避免以不完整資料產生誤導數字。",
            },
            "note": "目前無法取得新的 0050 融資維持率估算；不以舊資料冒充最新數值。",
        }

def main():
    now = datetime.now(TAIPEI)
    today = now.date().isoformat()
    previous_payload = {}
    try:
        if OUTPUT.exists():
            previous_payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    except Exception:
        previous_payload = {}
    previous_quotes = {
        item.get("ticker"): item for item in previous_payload.get("quotes", [])
        if item.get("ticker") and item.get("price") is not None
    }
    quotes = [fetch_quote(ticker, label, today) for ticker, label in WATCHLIST.items()]
    for quote in quotes:
        if quote.get("price") is None and quote.get("ticker") in previous_quotes:
            fallback = dict(previous_quotes[quote["ticker"]])
            fallback["status"] = "stale"
            fallback["is_stale_fallback"] = True
            fallback["message"] = "本次來源暫時無回傳；保留上次成功資料並明確標示時間，避免顯示空值。"
            quotes[quotes.index(quote)] = fallback
    old_margin = previous_payload.get("margin_maintenance", {})
    if not isinstance(old_margin, dict):
        old_margin = {}
    # Margin data is a daily close-based statistic. Avoid requesting the full
    # FinMind history on every 5-minute quote refresh. Refresh once after the
    # Taiwan cash session; reuse the snapshot for all other intraday cycles.
    if now.time().replace(tzinfo=None) < time(14, 0):
        margin_maintenance = old_margin or {
            "status": "unavailable",
            "history": [],
            "note": "0050 融資維持率將於台灣收盤後更新。",
        }
    elif old_margin.get("checked_date_taipei") == today:
        margin_maintenance = old_margin
    else:
        margin_maintenance = fetch_0050_margin_maintenance()
        margin_maintenance["checked_date_taipei"] = today
        if margin_maintenance.get("status") == "unavailable":
            old_etf = old_margin.get("etf_0050")
            if old_etf and old_margin.get("status") in ("estimated", "stale"):
                margin_maintenance["etf_0050"] = old_etf
                margin_maintenance["history"] = old_margin.get("history", [])
                margin_maintenance["status"] = "stale"
                margin_maintenance["previous_data_date"] = old_etf.get("date")
                margin_maintenance["note"] = "本次資料來源暫時無回傳；保留上次成功估算並標示原資料日期，不視為最新值。"
    etf = next((q for q in quotes if q["ticker"] == "0050.TW"), None)
    in_cash_hours = time(9, 0) <= now.time().replace(tzinfo=None) <= time(13, 35)
    etf_has_price = bool(etf and etf.get("price") is not None)
    etf_is_today = bool(etf and etf.get("status") == "live"
                        and str(etf.get("latest_bar_time", "")).startswith(today))
    if etf_is_today and in_cash_hours:
        status = "live"
        label = "台灣盤中行情更新中；美股行情依台灣時間持續更新"
    elif etf_has_price:
        status = "latest_data"
        label = f"台灣市場顯示最近有效交易日 {etf.get('market_date_taipei') or etf.get('latest_bar_time', '')[:10]}；美股期貨持續顯示最新可用資料"
    else:
        status = "partial_data"
        label = "部分市場資料暫缺；其餘標的仍顯示各自最新成功資料"
    payload = {
        "status": status,
        "label": label,
        "trading_date_taipei": today,
        "generated_at_taipei": now.isoformat(timespec="seconds"),
        "refresh_interval_minutes": 5,
        "market_hours_taipei": "09:30-13:30（台灣證券交易日）",
        "source": "Yahoo Finance 5-minute bars; indicative data, not an official exchange feed",
        "quotes": quotes,
        "margin_maintenance": margin_maintenance,
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
