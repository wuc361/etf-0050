"""Traceable quantitative explanations for 0050 forecasts."""
import numpy as np
import pandas as pd

def _label(name):
    key=name.removeprefix("f_")
    mapping={"0050_TW_ret1":"0050前一日報酬","0050_TW_ret5":"0050五日報酬","0050_rsi14":"0050 RSI(14)","0050_ma5_gap":"0050與MA5距離","0050_ma20_gap":"0050與MA20距離","0050_ma60_gap":"0050與MA60距離","0050_macd_hist":"0050 MACD動能","0050_volume_ratio20":"0050成交量／20日均量","0050_roc20":"0050二十日動能","IXIC_ret1":"NASDAQ前一日報酬","SOX_ret1":"費城半導體前一日報酬","GSPC_ret1":"S&P 500前一日報酬","TSM_ret1":"台積電ADR前一日報酬","NVDA_ret1":"NVIDIA前一日報酬","VIX_ret1":"VIX變化"}
    if key in mapping:return mapping[key]
    return key.replace("_"," ")

def explain_model_prediction(model,row,features,limit=8):
    try:
        imputer=model.named_steps["imputer"]; scaler=model.named_steps["scale"]; clf=model.named_steps["model"]
        z=scaler.transform(imputer.transform(row[features]))[0]
        names=list(imputer.get_feature_names_out(features)); values=z*clf.coef_[0]
        items=[]
        for name,value in zip(names,values):
            if name.endswith("_missing") or not np.isfinite(value) or abs(float(value))<1e-8:continue
            items.append({"feature":name,"label":_label(name),"contribution":round(float(value),5),"direction":"偏多" if value>0 else "偏空"})
        items.sort(key=lambda x:abs(x["contribution"]),reverse=True)
        return {"method":"標準化特徵值 × Logistic Regression 係數；正值推升上漲 log-odds，負值壓低上漲 log-odds。這是模型內部貢獻，不是因果證明。","top_positive":[x for x in items if x["contribution"]>0][:limit],"top_negative":[x for x in items if x["contribution"]<0][:limit],"all_ranked":items[:limit*2]}
    except Exception as exc:
        return {"method":"特徵貢獻計算失敗","error":str(exc),"top_positive":[],"top_negative":[]}

def build_market_analysis(briefing,history,drivers=None):
    bullish=[];bearish=[];neutral=[]
    def add(tone,item):
        (bullish if tone=="偏多" else bearish if tone=="偏空" else neutral).append(item)
    rows=briefing.get("us_market") or []; by={x.get("ticker"):x for x in rows}
    tech=[x for x in rows if x.get("ticker") in {"^IXIC","^SOX","TSM","NVDA","AMD","MU","ASML"} and x.get("change_pct") is not None]
    if tech:
        avg=float(np.mean([float(x["change_pct"]) for x in tech]))
        add("偏多" if avg>=.5 else "偏空" if avg<=-.5 else "中性",{"title":"美股科技／半導體組合","value":f"{avg:+.2f}% 平均漲跌","date":max(x.get("date","") for x in tech),"source":"Yahoo Finance 各標的最後完成交易時段","detail":"依 NASDAQ、SOX、台積電 ADR、NVIDIA、AMD、美光及 ASML 可用資料計算；非因果判定。"})
    for ticker,label in [("^GSPC","S&P 500"),("^VIX","VIX 波動率指數")]:
        x=by.get(ticker)
        if not x or x.get("change_pct") is None:continue
        v=float(x["change_pct"]); tone=("偏空" if v>=5 else "偏多" if v<=-5 else "中性") if ticker=="^VIX" else ("偏多" if v>=.3 else "偏空" if v<=-.3 else "中性")
        add(tone,{"title":label,"value":f"{v:+.2f}%","date":x.get("date"),"source":"Yahoo Finance","detail":"VIX 反映隱含波動與避險需求；S&P 500 為美股大型股方向參考。"})
    inst=briefing.get("institutional") or {}; total=inst.get("total_net_100m")
    if total is not None:
        v=float(total);add("偏多" if v>0 else "偏空" if v<0 else "中性",{"title":"台股三大法人合計","value":f"{'買超' if v>0 else '賣超' if v<0 else '接近平衡'} {abs(v):,.1f} 億元","date":inst.get("date"),"source":"TWSE BFI82U","detail":"全上市市場法人買賣金額，不等同0050單一ETF資金流。"})
    etf=briefing.get("etf_0050_institutional") or {}; total=etf.get("total_net_shares")
    if total is not None:
        v=int(total);add("偏多" if v>0 else "偏空" if v<0 else "中性",{"title":"0050個別ETF三大法人","value":f"{'買超' if v>0 else '賣超' if v<0 else '接近平衡'} {abs(v):,} 股","date":etf.get("date"),"source":"TWSE T86","detail":"0050個別證券法人買賣超股數。"})
    if history is not None and not history.empty:
        c=history["Close"].dropna().astype(float)
        if len(c)>=60:
            last=float(c.iloc[-1]);ma5=float(c.rolling(5).mean().iloc[-1]);ma20=float(c.rolling(20).mean().iloc[-1]);date=pd.Timestamp(c.index[-1]).date().isoformat()
            tone="偏多" if last>ma20 and ma5>ma20 else "偏空" if last<ma20 and ma5<ma20 else "中性"
            add(tone,{"title":"0050均線結構","value":f"收盤 {last:.2f}｜MA5 {ma5:.2f}｜MA20 {ma20:.2f}","date":date,"source":"0050.TW歷史收盤價","detail":"短期與月線趨勢是否同向；均線為落後指標。"})
            d=c.diff();g=d.clip(lower=0).rolling(14).mean().iloc[-1];l=(-d.clip(upper=0)).rolling(14).mean().iloc[-1]
            rsi=(100-100/(1+g/l)) if pd.notna(l) and l>0 else (100.0 if pd.notna(g) and g>0 else None)
            if rsi is not None:add("偏多" if rsi>=55 else "偏空" if rsi<=45 else "中性",{"title":"0050 RSI(14)","value":f"{rsi:.1f}","date":date,"source":"0050.TW歷史收盤價","detail":"高於55代表近期上漲動能較強，低於45代表下跌動能較強；不是單獨買賣訊號。"})
            macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();sig=macd.ewm(span=9,adjust=False).mean();h=float(macd.iloc[-1]-sig.iloc[-1])
            add("偏多" if h>0 else "偏空" if h<0 else "中性",{"title":"0050 MACD動能","value":f"{h:+.4f}","date":date,"source":"0050.TW歷史收盤價","detail":"柱狀體正值代表MACD線高於訊號線，負值則相反。"})
    for x in (drivers or {}).get("top_positive",[])[:4]:
        bullish.append({"title":"模型看漲貢獻："+x["label"],"value":f"log-odds {x['contribution']:+.3f}","date":None,"source":"Logistic Regression係數分解","detail":"此輸入值推升模型上漲傾向，不代表市場因果。"})
    for x in (drivers or {}).get("top_negative",[])[:4]:
        bearish.append({"title":"模型看跌貢獻："+x["label"],"value":f"log-odds {x['contribution']:+.3f}","date":None,"source":"Logistic Regression係數分解","detail":"此輸入值壓低模型上漲傾向，不代表市場因果。"})
    stance="偏多，但仍有反向風險" if len(bullish)>len(bearish) else "偏空，反彈訊號仍需確認" if len(bearish)>len(bullish) else "多空拉鋸，暫無明確優勢"
    return {"stance":stance,"summary":f"目前可追溯訊號：偏多 {len(bullish)} 項、偏空 {len(bearish)} 項、中性 {len(neutral)} 項。綜合判讀為「{stance}」。項目數不是勝率，且不同訊號權重不同，請檢視數值、日期及來源。","bullish_factors":bullish,"bearish_factors":bearish,"neutral_factors":neutral,"method":"可追溯規則式市場資料整理＋模型線性特徵貢獻；新聞標題不當作因果，也不直接覆寫量化機率。"}
