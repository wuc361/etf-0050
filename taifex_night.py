"""TAIFEX TX after-hours/night-session data collection and feature preparation.

The public DailyMarketReportFut endpoint is a latest-report feed, not a full
historical API. This module accumulates each fetched report in the repository.
Night-session rows are attributed to the following Taiwan trading date.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TAIFEX_API = "https://openapi.taifex.com.tw/v1/DailyMarketReportFut"
MIN_NIGHT_HISTORY = 120
FEATURE_NAMES = (
    "f_taifex_night_change_pct",
    "f_taifex_night_open_to_last_pct",
    "f_taifex_night_range_pct",
    "f_taifex_night_close_location",
    "f_taifex_night_volume_log",
)


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _field(row, *names):
    lookup = {_key(k): v for k, v in row.items()}
    for name in names:
        value = lookup.get(_key(name))
        if value not in (None, ""):
            return value
    return None


def _number(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    text = text.replace("▲", "").replace("▼", "").replace("+", "")
    if text in ("", "-", "--", "N/A", "nan", "None"):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _date(value):
    text = re.sub(r"[^0-9]", "", str(value or ""))
    if len(text) >= 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return None


def _session(value):
    text = str(value or "").strip().lower()
    if "盤後" in text or "after" in text or "night" in text:
        return "after_hours"
    if "一般" in text or "regular" in text or "day" in text:
        return "regular"
    return None


def _canonical(row):
    contract = str(_field(row, "Contract", "契約") or "").strip().upper()
    session = _session(_field(row, "TradingSession", "交易時段", "Session"))
    date_text = _date(_field(row, "Date", "日期", "交易日期"))
    month = str(_field(row, "ContractMonth(Week)", "ContractMonth", "到期月份") or "").strip()
    if contract != "TX" or not date_text or not session:
        return None
    return {
        "date": date_text,
        "contract": contract,
        "contract_month": month,
        "session": session,
        "open": _number(_field(row, "Open", "開盤價")),
        "high": _number(_field(row, "High", "最高價")),
        "low": _number(_field(row, "Low", "最低價")),
        "last": _number(_field(row, "Last", "Close", "最後成交價")),
        "change_points": _number(_field(row, "Change", "漲跌")),
        "change_pct": _number(_field(row, "%", "ChangePercent", "漲跌百分比")),
        "volume": _number(_field(row, "Volume", "成交量")),
        "settlement": _number(_field(row, "SettlementPrice", "Settlement Price", "結算價")),
        "open_interest": _number(_field(row, "OpenInterest", "未平倉量")),
        "source": "TAIFEX DailyMarketReportFut",
    }


def fetch_latest_report():
    request = urllib.request.Request(
        TAIFEX_API,
        headers={"User-Agent": "Mozilla/5.0 ETF0050Research/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data") or payload.get("Data") or payload.get("result") or []
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("TAIFEX API 回傳格式不是資料列清單")
    return rows


def update_taifex_history(path: Path):
    """Fetch latest official report and append/deduplicate TX day/night rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []

    errors = []
    fetched = []
    try:
        fetched = fetch_latest_report()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    by_key = {}
    for item in existing:
        if isinstance(item, dict) and item.get("date") and item.get("session"):
            key = (item["date"], item.get("contract", "TX"), item.get("contract_month", ""), item["session"])
            by_key[key] = item
    for raw in fetched:
        if not isinstance(raw, dict):
            continue
        item = _canonical(raw)
        if not item:
            continue
        key = (item["date"], item["contract"], item["contract_month"], item["session"])
        by_key[key] = item

    result = [by_key[k] for k in sorted(by_key)]
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, {
        "source": TAIFEX_API,
        "fetched_rows": len(fetched),
        "stored_rows": len(result),
        "updated_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "errors": errors[:3],
    }


def _near_month_rows(records, session="after_hours"):
    grouped = {}
    for row in records or []:
        if row.get("contract") != "TX" or row.get("session") != session:
            continue
        if row.get("last") is None:
            continue
        date_text = row.get("date")
        if not date_text:
            continue
        current = grouped.get(date_text)
        month = str(row.get("contract_month") or "")
        if current is None or (month and month < str(current.get("contract_month") or "999999")):
            grouped[date_text] = row
    return grouped


def _feature_values(row):
    if not row:
        return {}
    open_price, high, low, last = (row.get(k) for k in ("open", "high", "low", "last"))
    volume = row.get("volume")
    change_pct = row.get("change_pct")
    values = {}
    if change_pct is not None:
        values[FEATURE_NAMES[0]] = float(change_pct) / 100.0
    if open_price and last is not None:
        values[FEATURE_NAMES[1]] = float(last) / float(open_price) - 1.0
    if high is not None and low is not None and last:
        values[FEATURE_NAMES[2]] = (float(high) - float(low)) / abs(float(last))
        spread = float(high) - float(low)
        if spread > 0:
            values[FEATURE_NAMES[3]] = (float(last) - float(low)) / spread
    if volume is not None and volume >= 0:
        values[FEATURE_NAMES[4]] = math.log1p(float(volume))
    return values


def build_taifex_feature_map(records):
    """Return date-indexed, completed night-session features for supervised rows."""
    grouped = _near_month_rows(records, "after_hours")
    return {date_text: _feature_values(row) for date_text, row in grouped.items()}


def night_feature_values(row):
    return _feature_values(row)


def latest_night_for_prediction(records, latest_taiwan_date):
    """Only return a night session attributed to a date after the latest cash close."""
    candidates = [
        row for row in _near_month_rows(records, "after_hours").values()
        if row.get("date", "") > latest_taiwan_date
    ]
    return max(candidates, key=lambda row: row["date"]) if candidates else None


def build_night_dashboard(records, fetch_status, latest_taiwan_date):
    nights = _near_month_rows(records, "after_hours")
    latest = max(nights.values(), key=lambda row: row["date"]) if nights else None
    fresh = latest_night_for_prediction(records, latest_taiwan_date)
    history_count = len(nights)
    if latest is None:
        label = "尚未取得台指期夜盤資料"
    elif fresh:
        label = "夜盤已收盤，可供下一交易日開盤模型使用"
    else:
        label = "已保存最近夜盤；等待下一交易日所屬夜盤收盤"
    model_enabled = history_count >= MIN_NIGHT_HISTORY
    return {
        "status": "available" if latest else "unavailable",
        "label": label,
        "latest": latest,
        "prediction_session": fresh,
        "history_sessions": history_count,
        "model_features_enabled": model_enabled,
        "model_feature_status": (
            f"已累積 {history_count} 個夜盤交易日，夜盤特徵已納入模型"
            if model_enabled else
            f"夜盤歷史累積 {history_count}/{MIN_NIGHT_HISTORY} 個交易日；樣本不足，暫不加入模型機率"
        ),
        "fetch": fetch_status or {},
        "source": TAIFEX_API,
        "attribution_note": "TAIFEX 夜盤交易日歸屬隔日交易日；模型只使用台股開盤前已完成的夜盤資料。",
    }
