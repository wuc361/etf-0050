"""RSS news context for the 0050 dashboard; keyword heuristic, not a trained model."""
from __future__ import annotations
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

FEEDS = [
    ("國際市場", "美股 聯準會 利率 通膨 Nasdaq 半導體", "zh-TW", "TW", "TW"),
    ("國際市場", "Federal Reserve interest rates inflation Nasdaq semiconductor stocks", "en-US", "US", "US"),
    ("產業消息", "台積電 AI 晶片 NVIDIA 半導體", "zh-TW", "TW", "TW"),
    ("台灣市場", "台股 加權指數 外資 台灣股市", "zh-TW", "TW", "TW"),
    ("台灣市場", "台灣央行 新台幣 出口 景氣", "zh-TW", "TW", "TW"),
]
POSITIVE = ["上調","升評","獲利成長","營收創高","創新高","需求強勁","訂單增加","降息","通膨降溫","優於預期","買超","上漲","成長","利多","回升","upgrade","beat estimates","strong demand","rate cut","rally","surge","growth"]
NEGATIVE = ["下調","降評","獲利衰退","營收下滑","需求疲弱","訂單減少","升息","通膨升溫","低於預期","賣超","下跌","衰退","利空","制裁","關稅","戰爭","地緣政治","暴跌","downgrade","miss estimates","weak demand","rate hike","selloff","recession","tariff","sanctions","war"]

def _clean(value):
    return re.sub(r"<[^>]+>", "", value or "").strip()

def _score(title):
    low = title.lower()
    pos = [w for w in POSITIVE if w.lower() in low]
    neg = [w for w in NEGATIVE if w.lower() in low]
    return max(-2, min(2, len(pos)-len(neg))), pos[:3], neg[:3]

def _fetch(category, query, hl, gl, ceid):
    params = urllib.parse.urlencode({"q":query,"hl":hl,"gl":gl,"ceid":f"{ceid}:{hl.split('-')[0]}"})
    url = "https://news.google.com/rss/search?" + params
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 ETF0050Research/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        root = ET.fromstring(response.read())
    items = []
    for item in root.findall(".//item")[:12]:
        title = _clean(item.findtext("title",""))
        link = item.findtext("link","")
        source_node = item.find("source")
        source = _clean(source_node.text if source_node is not None else "Google News")
        try:
            published = parsedate_to_datetime(item.findtext("pubDate","")).astimezone(timezone.utc).isoformat(timespec="minutes")
        except Exception:
            published = ""
        score, positive, negative = _score(title)
        if title and link:
            items.append({"category":category,"title":title,"url":link,"source":source,"published_at":published,"keyword_score":score,"positive_terms":positive,"negative_terms":negative})
    return items

def collect_news(max_items=18):
    rows, errors = [], []
    for args in FEEDS:
        try:
            rows.extend(_fetch(*args))
        except Exception as exc:
            errors.append(f"{args[0]}新聞來源讀取失敗：{type(exc).__name__}")
    unique = {}
    for item in rows:
        key = re.sub(r"\W+", "", item["title"].lower())
        if key not in unique or item["published_at"] > unique[key]["published_at"]:
            unique[key] = item
    items = sorted(unique.values(), key=lambda x:x["published_at"], reverse=True)[:max_items]
    def group(category):
        selected = [x for x in items if x["category"] == category]
        score = sum(x["keyword_score"] for x in selected)
        tone = "偏多" if score > 0 else "偏空" if score < 0 else "中性或訊號不足"
        return {"tone":tone,"keyword_score":score,"article_count":len(selected),"items":selected[:6]}
    return {
        "generated_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method":"Google News RSS 標題關鍵字初步分類；非語意模型、非投資建議",
        "international":group("國際市場"),
        "domestic":group("台灣市場"),
        "industry":group("產業消息"),
        "items":items,
        "source_errors":errors,
        "limitations":[
            "僅分析 RSS 新聞標題，不代表已閱讀全文或核實事件。",
            "關鍵字可能誤判語境、反諷及利多出盡等情況。",
            "新聞目前是獨立情境參考，尚未納入歷史訓練及回測，不會改動模型預測機率。"
        ]
    }
