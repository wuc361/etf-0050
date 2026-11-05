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

def fetch_market_margin_maintenance() -> dict:
    """Fetch the latest published third-party market-wide maintenance estimate."""
    url = "https://www.istock.tw/post/twmarginrequirement"
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ETF-0050-monitor/1.0)"})
        with urlopen(request, timeout=18) as response:
            raw = response.read().decode("utf-8", errors="replace")
        text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
        text = re.sub(r"\s+", " ", text)
        match = re.search(
            r"最新大盤融資維持率.*?上市[）)]?\s*([0-9]+(?:\.[0-9]+)?)%.*?(20\d{2}-\d{2}-\d{2}).*?上櫃\s*([0-9]+(?:\.[0-9]+)?)%.*?整體\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
        )
        if not match:
            raise ValueError("頁面未找到上市、上櫃及整體融資維持率")
        listed, date, otc, overall = match.groups()
        return {
            "status": "estimated",
            "data_date": date,
            "listed_pct": float(listed),
            "otc_pct": float(otc),
            "overall_pct": float(overall),
            "source": "愛玩股（彙整市場公開資料）",
            "source_url": url,
            "is_official_aggregate": False,
            "note": "第三方彙整／推估的市場平均值，不是證交所逐戶整戶維持率，也不是個人帳戶數字。",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "data_date": None,
            "listed_pct": None,
            "otc_pct": None,
            "overall_pct": None,
            "source": "愛玩股",
            "source_url": url,
            "is_official_aggregate": False,
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            "note": "目前無法取得最新估算值；請勿以舊資料冒充最新維持率。",
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
    margin_maintenance = fetch_market_margin_maintenance()
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
