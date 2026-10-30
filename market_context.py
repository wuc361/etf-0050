"""Official TWSE institutional-flow history and previous-session market briefing."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TWSE_URL = "https://www.twse.com.tw/rwd/zh/fund/BFI82U"
TWSE_REPORT_URL = "https://www.twse.com.tw/zh/fund/BFI82U?response=html&type=day"
US_TICKERS = {
    "^GSPC": "S&P 500", "^IXIC": "NASDAQ 綜合指數",
    "^SOX": "費城半導體指數", "^DJI": "道瓊指數",
    "TSM": "台積電 ADR", "NVDA": "NVIDIA", "AMD": "AMD",
    "MU": "美光", "ASML": "ASML", "^VIX": "VIX 波動率指數",
    "^TNX": "美國10年期公債殖利率", "^FVX": "美國5年期公債殖利率"
}


def _request_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ETF0050Research/1.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_twse_institutional(date_text: str):
    """Fetch official BFI82U data; values are NTD, returned in NTD 100m."""
    query = urllib.parse.urlencode({
        "response": "json", "dayDate": date_text.replace("-", ""), "type": "day"
    })
    payload = _request_json(TWSE_URL + "?" + query)
    if str(payload.get("stat", "")).upper() not in {"OK", "OK "}:
        return None
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    if not fields or not rows:
        return None
    records = []
    for row in rows:
        if len(row) < 4:
            continue
        name = str(row[0]).strip()
        try:
            buy, sell, net = [int(str(v).replace(",", "").strip()) for v in row[1:4]]
        except (TypeError, ValueError):
            continue
        records.append((name, buy, sell, net))
    if not records:
        return None

    def sum_net(keys):
        return sum(net for name, _, _, net in records if any(key in name for key in keys))

    foreign = sum_net(["外資及陸資(不含外資自營商)"])
    trust = sum_net(["投信"])
    dealer = sum_net(["自營商(自行買賣)", "自營商(避險)"])
    total = sum_net(["合計"])
    if not total:
        total = foreign + trust + dealer
    return {
        "date": date_text,
        "foreign_net_ntd": foreign,
        "trust_net_ntd": trust,
        "dealer_net_ntd": dealer,
        "total_net_ntd": total,
        "foreign_net_100m": round(foreign / 1e8, 3),
        "trust_net_100m": round(trust / 1e8, 3),
        "dealer_net_100m": round(dealer / 1e8, 3),
        "total_net_100m": round(total / 1e8, 3),
        "source": "TWSE BFI82U",
        "source_url": f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=html&dayDate={date_text.replace('-', '')}&type=day"
    }


def fetch_twse_0050_flow(date_text: str):
    """Fetch 0050's own institutional net shares from the official TWSE T86 report."""
    query = urllib.parse.urlencode({
        "response": "json", "date": date_text.replace("-", ""),
        "selectType": "ALLBUT0999"
    })
    payload = _request_json("https://www.twse.com.tw/rwd/zh/fund/T86?" + query)
    if str(payload.get("stat", "")).upper() not in {"OK", "OK "}:
        return None
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    if not fields or not rows:
        return None
    try:
        code_idx = fields.index("證券代號")
        foreign_idx = fields.index("外陸資買賣超股數(不含外資自營商)")
        trust_idx = fields.index("投信買賣超股數")
        dealer_idx = fields.index("自營商買賣超股數")
        total_idx = fields.index("三大法人買賣超股數")
    except ValueError:
        return None
    for row in rows:
        if len(row) <= max(code_idx, foreign_idx, trust_idx, dealer_idx, total_idx):
            continue
        if str(row[code_idx]).strip() != "0050":
            continue
        def integer(value):
            return int(str(value).replace(",", "").strip())
        return {
            "0050_foreign_net_shares": integer(row[foreign_idx]),
            "0050_trust_net_shares": integer(row[trust_idx]),
            "0050_dealer_net_shares": integer(row[dealer_idx]),
            "0050_total_net_shares": integer(row[total_idx]),
            "0050_flow_source": "TWSE T86",
            "0050_flow_source_url": f"https://www.twse.com.tw/rwd/zh/fund/T86?response=html&date={date_text.replace('-', '')}&selectType=ALLBUT0999"
        }
    return None

def update_institutional_history(trading_dates, path: Path, backfill_sessions: int = 60, max_fetch: int = 10):
    """Backfill recent sessions once, then append each new session to the committed JSON cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        existing = []
    by_date = {x.get("date"): x for x in existing if x.get("date")}
    dates = [d.date().isoformat() for d in trading_dates[-backfill_sessions:]]
    missing = [d for d in dates if d not in by_date or "0050_total_net_shares" not in by_date[d]]
    # Always prioritize the latest Taiwan session, then backfill older dates in small batches.
    latest_date = dates[-1] if dates else None
    selected = ([latest_date] if latest_date in missing else [])
    selected += [d for d in missing if d != latest_date][:max(0, max_fetch - len(selected))]
    errors = []
    for date_text in selected:
        try:
            record = by_date.get(date_text, {})
            if not record:
                record = fetch_twse_institutional(date_text) or {}
            try:
                security_flow = fetch_twse_0050_flow(date_text)
                if security_flow:
                    record.update(security_flow)
                else:
                    errors.append(f"{date_text}: T86找不到0050資料")
            except Exception as exc:
                errors.append(f"{date_text}: T86 {type(exc).__name__}")
            if record:
                by_date[date_text] = record
            else:
                errors.append(f"{date_text}: BFI82U無資料或非交易日")
        except Exception as exc:
            errors.append(f"{date_text}: {type(exc).__name__}")
        time.sleep(0.12)
    result = [by_date[k] for k in sorted(by_date)]
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, errors


def add_institutional_features(data, records):
    import pandas as pd
    if not records:
        for name in ("foreign", "trust", "dealer", "total"):
            data[f"f_twse_{name}_net_100m"] = float("nan")
            data[f"f_twse_{name}_net_5d_100m"] = float("nan")
        return data
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    for name in ("foreign", "trust", "dealer", "total"):
        col = f"{name}_net_100m"
        if col not in frame:
            continue
        aligned = frame[col].reindex(data.index).astype(float)
        data[f"f_twse_{name}_net_100m"] = aligned.shift(1)
        data[f"f_twse_{name}_net_5d_100m"] = aligned.rolling(5).sum().shift(1)
    for name in ("foreign", "trust", "dealer", "total"):
        col = f"0050_{name}_net_shares"
        if col in frame:
            aligned = frame[col].reindex(data.index).astype(float) / 1000.0
            data[f"f_twse_0050_{name}_net_1000shares"] = aligned.shift(1)
            data[f"f_twse_0050_{name}_net_5d_1000shares"] = aligned.rolling(5).sum().shift(1)
    return data


def latest_institutional_features(records, latest_taiwan_date):
    matches = [x for x in records if x.get("date") == latest_taiwan_date]
    if not matches:
        return {}
    row = matches[-1]
    result = {}
    for name in ("foreign", "trust", "dealer", "total"):
        result[f"f_twse_{name}_net_100m"] = float(row.get(f"{name}_net_100m", float("nan")))
        history = [x for x in records if x.get("date", "") <= latest_taiwan_date][-5:]
        values = [float(x[f"{name}_net_100m"]) for x in history if x.get(f"{name}_net_100m") is not None]
        result[f"f_twse_{name}_net_5d_100m"] = sum(values) if values else float("nan")
    for name in ("foreign", "trust", "dealer", "total"):
        result[f"f_twse_0050_{name}_net_1000shares"] = float(row.get(f"0050_{name}_net_shares", float("nan"))) / 1000.0
        history = [x for x in records if x.get("date", "") <= latest_taiwan_date][-5:]
        values = [float(x[f"0050_{name}_net_shares"]) / 1000.0 for x in history if x.get(f"0050_{name}_net_shares") is not None]
        result[f"f_twse_0050_{name}_net_5d_1000shares"] = sum(values) if values else float("nan")
    return result


def _last_completed_us_session():
    now = datetime.now(ZoneInfo("America/New_York"))
    # A daily bar is not considered final until after the regular 16:00 close.
    cutoff_date = now.date() if now.hour > 16 or (now.hour == 16 and now.minute >= 15) else now.date() - timedelta(days=1)
    return cutoff_date.isoformat(), now.isoformat(timespec="minutes")


def build_market_briefing(raw, institutional_records, news, taiwan_data_date, institutional_errors=None):
    us_cutoff, ny_now = _last_completed_us_session()
    us_rows = []
    for ticker, label in US_TICKERS.items():
        frame = raw.get(ticker)
        if frame is None or "Close" not in frame:
            continue
        series = frame["Close"].dropna()
        eligible = series[series.index.date.astype(str) <= us_cutoff]
        if len(eligible) < 2:
            continue
        last_date = eligible.index[-1].date().isoformat()
        previous = float(eligible.iloc[-2])
        latest = float(eligible.iloc[-1])
        change = latest / previous - 1 if previous else 0.0
        is_yield = ticker in {"^TNX", "^FVX"}
        us_rows.append({
            "ticker": ticker, "name": label, "date": last_date,
            "close": latest, "previous_close": previous,
            "display_close": f"{latest / 10:.2f}%" if is_yield else f"{latest:,.2f}",
            "display_change": f"{(latest - previous) * 10:+.1f} bps" if is_yield else f"{change * 100:+.2f}%",
            "change_pct": round(change * 100, 3),
            "yield_change_bps": round((latest - previous) * 10, 2) if is_yield else None,
            "tone": (
                ("波動風險升高" if change > 0 else "波動風險下降" if change < 0 else "中性")
                if ticker == "^VIX" else
                ("殖利率上升" if latest > previous else "殖利率下降" if latest < previous else "中性")
                if is_yield else
                ("偏多" if change > 0 else "偏空" if change < 0 else "中性")
            )
        })
    actual_us_dates = [x["date"] for x in us_rows]
    us_session_date = max(actual_us_dates) if actual_us_dates else us_cutoff
    institutional = next((x for x in institutional_records if x.get("date") == taiwan_data_date), None)

    events = []
    if institutional:
        for key, label in [
            ("foreign_net_100m", "外資"),
            ("trust_net_100m", "投信"),
            ("dealer_net_100m", "自營商")
        ]:
            amount = float(institutional.get(key, 0))
            events.append({
                "group": "台灣法人籌碼", "title": f"{label}{'買超' if amount > 0 else '賣超' if amount < 0 else '買賣平衡'} {abs(amount):,.1f} 億元",
                "date": taiwan_data_date, "tone": "偏多" if amount > 0 else "偏空" if amount < 0 else "中性",
                "detail": "TWSE 三大法人買賣金額統計表；為全市場金額，不等於0050單一ETF的資金流。"
            })
        total = float(institutional.get("total_net_100m", 0))
        events.append({
            "group": "台灣法人籌碼", "title": f"三大法人合計{'買超' if total > 0 else '賣超' if total < 0 else '買賣平衡'} {abs(total):,.1f} 億元",
            "date": taiwan_data_date, "tone": "偏多" if total > 0 else "偏空" if total < 0 else "中性",
            "detail": "外資、投信及自營商合計；自營商自行買賣與避險部位合併計算。"
        })
        if "0050_total_net_shares" in institutional:
            for key, label in [
                ("0050_foreign_net_shares", "0050外資"),
                ("0050_trust_net_shares", "0050投信"),
                ("0050_dealer_net_shares", "0050自營商")
            ]:
                shares = int(institutional.get(key, 0))
                events.append({
                    "group": "0050個別ETF籌碼",
                    "title": f"{label}{'買超' if shares > 0 else '賣超' if shares < 0 else '買賣平衡'} {abs(shares):,} 股",
                    "date": taiwan_data_date,
                    "tone": "偏多" if shares > 0 else "偏空" if shares < 0 else "中性",
                    "detail": "TWSE T86 0050個別證券三大法人買賣超日報；單位為股。"
                })
    else:
        events.append({
            "group": "台灣法人籌碼", "title": "尚未取得前一台灣交易日法人資料",
            "date": taiwan_data_date, "tone": "資料不足",
            "detail": "TWSE BFI82U 未回傳有效資料；不以其他日期資料代替。"
        })

    equity_tickers = {"^GSPC", "^IXIC", "^SOX", "^DJI", "TSM", "NVDA", "AMD", "MU", "ASML"}
    us_equity_rows = [x for x in us_rows if x["ticker"] in equity_tickers]
    us_negative = [x for x in us_equity_rows if x["change_pct"] <= -1.5]
    us_positive = [x for x in us_equity_rows if x["change_pct"] >= 1.5]
    if us_negative:
        names = "、".join(x["name"] for x in us_negative)
        events.append({"group":"美股前夜","title":f"美股市場出現較大跌幅：{names}","date":us_session_date,"tone":"偏空","detail":"單日跌幅達或超過1.5%的觀察標的；此為量化門檻，不代表事件因果。"})
    if us_positive:
        names = "、".join(x["name"] for x in us_positive)
        events.append({"group":"美股前夜","title":f"美股市場出現較大漲幅：{names}","date":us_session_date,"tone":"偏多","detail":"單日漲幅達或超過1.5%的觀察標的；此為量化門檻，不代表事件因果。"})

    news_items = (news or {}).get("items", [])
    # Keep recent items for context; exact article timestamps remain visible in the UI.
    recent = []
    now_utc = datetime.now(timezone.utc)
    for item in news_items:
        try:
            published = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
            age_hours = (now_utc - published).total_seconds() / 3600
            if -1 <= age_hours <= 48:
                item_view = dict(item)
                item_view["published_at_taipei"] = published.astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M 台北時間")
                recent.append(item_view)
        except Exception:
            continue
    us_news = [x for x in recent if x.get("category") == "國際市場"][:8]
    domestic_news = [x for x in recent if x.get("category") == "台灣市場"][:8]
    industry_news = [x for x in recent if x.get("category") == "產業消息"][:8]
    for item in us_news[:5]:
        events.append({"group":"國際新聞","title":item.get("title",""),"date":item.get("published_at_taipei",item.get("published_at","")),"tone":"偏空" if item.get("keyword_score",0)<0 else "偏多" if item.get("keyword_score",0)>0 else "待判讀","detail":item.get("source","新聞來源")+"｜可能影響：美股風險偏好、利率預期與科技股估值；請點開全文核對。","url":item.get("url","")})
    for item in domestic_news[:4]:
        events.append({"group":"台灣新聞","title":item.get("title",""),"date":item.get("published_at",""),"tone":"偏空" if item.get("keyword_score",0)<0 else "偏多" if item.get("keyword_score",0)>0 else "待判讀","detail":item.get("source","新聞來源")+"｜可能影響：台灣政策、匯率、資金面及本地市場風險偏好；請點開全文核對。","url":item.get("url","")})
    for item in industry_news[:4]:
        events.append({"group":"產業新聞","title":item.get("title",""),"date":item.get("published_at",""),"tone":"偏空" if item.get("keyword_score",0)<0 else "偏多" if item.get("keyword_score",0)>0 else "待判讀","detail":item.get("source","新聞來源")+"｜可能影響：台積電、AI伺服器與半導體供應鏈等0050權值成分股；請點開全文核對。","url":item.get("url","")})

    total = float(institutional.get("total_net_100m", 0)) if institutional else None
    us_tech = [x for x in us_rows if x["ticker"] in {"^IXIC","^SOX","TSM","NVDA","AMD","MU","ASML"}]
    avg_tech = sum(x["change_pct"] for x in us_tech) / len(us_tech) if us_tech else None
    if avg_tech is None:
        narrative = "美股前夜資料不足，暫不對隔日台股方向做事件推論。"
    elif avg_tech <= -1.0:
        narrative = "美股科技與半導體觀察組合偏弱，可能對台灣電子權值及0050開盤情緒形成壓力；仍需搭配台灣法人籌碼與匯率確認。"
    elif avg_tech >= 1.0:
        narrative = "美股科技與半導體觀察組合偏強，可能對台灣電子權值及0050開盤情緒形成支撐；仍需搭配台灣法人籌碼與匯率確認。"
    else:
        narrative = "美股科技與半導體觀察組合變動有限，暫未形成明確單向訊號。"
    if total is not None:
        narrative += f" 前一台灣交易日三大法人合計{'買超' if total>0 else '賣超' if total<0 else '接近平衡'} {abs(total):,.1f} 億元。"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taiwan_data_date": taiwan_data_date,
        "us_session_date": us_session_date,
        "us_timezone_checked_at": ny_now,
        "us_market": us_rows,
        "institutional": institutional,
        "etf_0050_institutional": ({
            "date": institutional.get("date"),
            "foreign_net_shares": institutional.get("0050_foreign_net_shares"),
            "trust_net_shares": institutional.get("0050_trust_net_shares"),
            "dealer_net_shares": institutional.get("0050_dealer_net_shares"),
            "total_net_shares": institutional.get("0050_total_net_shares"),
            "source_url": institutional.get("0050_flow_source_url")
        } if institutional and "0050_total_net_shares" in institutional else None),
        "institutional_errors": (institutional_errors or [])[:8],
        "events": events[:24],
        "us_news": us_news,
        "domestic_news": domestic_news,
        "industry_news": industry_news,
        "narrative": narrative,
        "interpretation_note": "事件清單是可查證資料與規則式提示，不是因果判定；新聞標題關鍵字只供人工閱讀，不直接改寫模型機率。",
        "sources": [
            {"name":"TWSE 三大法人買賣金額統計表 BFI82U","url":TWSE_REPORT_URL,"role":"全市場法人買賣金額"},
            {"name":"TWSE 三大法人買賣超日報 T86","url":"https://www.twse.com.tw/zh/trading/foreign/t86.html","role":"0050個別ETF法人買賣超股數"},
            {"name":"TWSE OpenAPI","url":"https://openapi.twse.com.tw/","role":"上市公司重大訊息與市場資料"},
            {"name":"TAIFEX","url":"https://www.taifex.com.tw/","role":"台指期及法人未平倉"},
            {"name":"FRED","url":"https://fred.stlouisfed.org/","role":"美國公債殖利率與總體資料"},
            {"name":"CBOE VIX","url":"https://www.cboe.com/tradable_products/vix/","role":"市場隱含波動率"}
        ]
    }


def sync_official_0050_latest(raw: dict, max_staleness_days: int = 7):
    """Use the official TWSE latest daily OHLCV bar for 0050; never model on a stale Yahoo bar."""
    import pandas as pd

    now_taipei = datetime.now(ZoneInfo("Asia/Taipei"))
    month_start = now_taipei.replace(day=1).strftime("%Y%m%d")
    query = urllib.parse.urlencode({
        "response": "json", "date": month_start, "stockNo": "0050"
    })
    payload = _request_json("https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?" + query)
    rows = payload.get("data") or []
    parsed = []
    for row in rows:
        try:
            raw_date = str(row[0]).replace("/", "").strip()
            if len(raw_date) != 7:
                continue
            yyyy = int(raw_date[:3]) + 1911
            date_iso = f"{yyyy:04d}-{raw_date[3:5]}-{raw_date[5:7]}"
            values = [float(str(v).replace(",", "").strip()) for v in row[1:7]]
            volume, turnover, open_price, high_price, low_price, close_price = values
            parsed.append({
                "date": date_iso, "volume": volume, "turnover": turnover,
                "open": open_price, "high": high_price, "low": low_price,
                "close": close_price
            })
        except (ValueError, TypeError, IndexError):
            continue
    if not parsed:
        raise RuntimeError("TWSE官方0050日成交資訊無有效OHLCV資料，停止發布預測。")

    official = max(parsed, key=lambda x: x["date"])
    frame = raw.get("0050.TW")
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo Finance沒有0050歷史資料，無法建立完整模型序列。")
    frame = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    yahoo_date = frame.index.max().date().isoformat()
    official_date = official["date"]
    gap = (pd.Timestamp(official_date) - pd.Timestamp(yahoo_date)).days
    if gap < 0:
        raise RuntimeError(
            f"Yahoo資料日期 {yahoo_date} 晚於TWSE官方最新日期 {official_date}；日期狀態異常，停止發布。"
        )
    if gap > max_staleness_days:
        raise RuntimeError(
            f"Yahoo資料落後TWSE官方資料超過{max_staleness_days}個日曆日（Yahoo={yahoo_date}、TWSE={official_date}）；"
            "缺少中間交易日，停止發布以免錯誤補值。"
        )

    bar = pd.DataFrame([{
        "Open": official["open"], "High": official["high"],
        "Low": official["low"], "Close": official["close"],
        "Volume": official["volume"]
    }], index=[pd.Timestamp(official_date)])
    # Official TWSE is the source of truth for the latest bar. Replace a same-date
    # Yahoo row or append the missing latest Taiwan session.
    frame = pd.concat([frame, bar])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    raw["0050.TW"] = frame
    return raw, {
        "date": official_date,
        "close": official["close"],
        "open": official["open"],
        "high": official["high"],
        "low": official["low"],
        "volume": int(official["volume"]),
        "source": "TWSE STOCK_DAY",
        "yahoo_previous_latest_date": yahoo_date,
        "yahoo_lag_calendar_days": gap,
        "status": "official_bar_merged"
    }
