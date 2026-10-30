import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.data import load_market_data
from src.features import build_features, FACTOR_REGISTRY
from src.model import walk_forward_evaluate, fit_latest_model, save_report

st.set_page_config(page_title="0050 AI 預測中心", page_icon="📈", layout="wide")
st.title("0050 AI 預測中心")
st.caption("V1.0｜開盤前方向預測｜五年歷史回測｜雲端部署版起始專案")

with st.sidebar:
    st.header("模型設定")
    years = st.selectbox("歷史資料期間", [3, 5, 10], index=1)
    test_window = st.selectbox("每次樣本外測試交易日", [20, 40, 60], index=1)
    st.info("預測目標：當日收盤價相對前一交易日收盤價上漲或下跌。")
    run = st.button("更新資料並執行回測", type="primary", use_container_width=True)

st.warning("目前是可部署的 V1 起始版。尚未連接雲端資料庫、正式新聞情緒 API 或付費即時行情；畫面不會虛構即時預測或 90% 準確率。")

if run or "data" not in st.session_state:
    with st.spinner("下載市場資料並建立特徵…"):
        try:
            raw = load_market_data(years=years)
            data = build_features(raw)
            st.session_state["data"] = data
            st.session_state["raw"] = raw
            st.session_state["years"] = years
        except Exception as e:
            st.error(f"資料更新失敗：{e}")
            st.stop()

data = st.session_state.get("data")
if data is not None and len(data):
    latest = data.dropna(subset=["target"]).iloc[-1] if data["target"].notna().any() else data.iloc[-1]
    a,b,c,d = st.columns(4)
    a.metric("可用交易日", f"{len(data):,}")
    b.metric("可用模型因子", f"{len([x for x in data.columns if x.startswith('f_')])}")
    c.metric("資料最新日期", str(data.index.max().date()))
    d.metric("模型狀態", "待驗證")

    st.subheader("歷史資料")
    chart_cols = [c for c in ["0050.TW", "2330.TW", "TSM", "^SOX", "^IXIC"] if c in st.session_state["raw"]]
    if "0050.TW" in st.session_state["raw"]:
        st.line_chart(st.session_state["raw"]["0050.TW"]["Close"], height=250)
    st.caption("價格圖為資料檢查用途，不是投資建議。")

    st.subheader("樣本外滾動回測")
    if st.button("開始 Walk-forward 回測"):
        with st.spinner("逐段訓練及測試，請稍候…"):
            report, predictions = walk_forward_evaluate(data, test_window=test_window)
            st.session_state["report"] = report
            st.session_state["predictions"] = predictions
            save_report(report, predictions)
    if "report" in st.session_state:
        r = st.session_state["report"]
        x,y,z = st.columns(3)
        x.metric("樣本外方向準確率", f"{r['accuracy']:.1%}" if pd.notna(r["accuracy"]) else "N/A")
        y.metric("上漲命中率", f"{r['up_recall']:.1%}" if pd.notna(r["up_recall"]) else "N/A")
        z.metric("預測覆蓋率", f"{r['coverage']:.1%}" if pd.notna(r["coverage"]) else "N/A")
        st.write(f"測試筆數：{r['n_test']}；基準上漲比例：{r['base_rate']:.1%}")
        st.dataframe(st.session_state["predictions"].tail(30), use_container_width=True)
        if r["accuracy"] < .90:
            st.info("目前未達 90% 目標。不得以訓練集分數替代樣本外準確率。")
    st.subheader("50 項候選因子清單")
    st.dataframe(pd.DataFrame(FACTOR_REGISTRY), use_container_width=True, hide_index=True)
else:
    st.info("尚無資料。請按左側按鈕更新。")

st.caption("免責聲明：研究與教育用途。歷史績效不代表未來結果；模型輸出不構成投資建議。")
