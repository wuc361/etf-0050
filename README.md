# 0050 AI Forecast

以 GitHub Actions 執行每日資料更新、樣本外回測與下一交易日方向研究；GitHub Pages 提供靜態預測儀表板，不需要 Streamlit 常駐服務。

## 架構
- `data.py`：下載研究用市場價格資料（Yahoo Finance，非官方來源）。
- `features.py`：產生1/2/3交易日預測標籤、跨市場報酬、MA60、RSI、MACD、布林通道、ATR、量能等特徵。
- `model.py`：Logistic Regression 基準模型、Purged expanding-window Walk-forward 回測、Balanced Accuracy、Brier Score及多數類別基準比較。
- `run_daily.py`：由 GitHub Actions 呼叫，輸出 `site/data/latest.json`。
- `news.py`：讀取 Google News RSS，整理國際市場、台灣市場與半導體產業新聞標題，提供來源連結及初步關鍵字情緒。
- `site/index.html`：不依賴 Streamlit 的靜態儀表板，顯示三種預測期限、K線/均線/布林通道/成交量、RSI、MACD、回測品質及指標教學。
- `.github/workflows/daily.yml`：排程執行並部署 GitHub Pages。

## 啟用方式
1. 在 GitHub Repository 的 **Settings → Actions → General → Workflow permissions** 確認工作流程允許必要的讀寫權限。
2. 在 **Settings → Pages** 將 **Build and deployment → Source** 設為 **GitHub Actions**。
3. 前往 **Actions → 0050 AI Daily Forecast → Run workflow**，手動執行第一次流程。
4. 成功後從 **Settings → Pages** 查看網站網址。排程為星期一至星期五 UTC 00:30（台灣時間 08:30），但 GitHub 排程可能延遲，且不是交易日曆。
5. 若工作流程失敗，開啟 **Actions** 查看失敗步驟與錯誤紀錄。

## 研究限制
- Yahoo Finance 是非官方資料來源，行情完整性需自行交叉驗證；TWSE、TAIFEX、FRED、CBOE及元大投信連結目前提供查證入口，並非全部已自動接入模型。
- 目前使用 Logistic Regression 基準分類器，不保證準確率或獲利。回測同時提供 Balanced Accuracy、Brier Score與多數類別基準，請避免只看 Accuracy。
- 跨市場時區與交易日曆尚未完整建模；盤前可用性需要進一步驗證。
- 新聞目前以 RSS 標題關鍵字作初步分類，不代表全文語意分析或事件查證；不會直接改動模型機率，也尚未納入歷史回測。
- 2日、3日預測為相對最新收盤價的累積方向分類，不是逐日價格路徑；各期限分開訓練及回測，並在回測切分時排除可能重疊的未來標籤。
- K線為 Yahoo Finance 調整後日線資料，最多顯示最近120個交易日；不等同即時報價。
- 回測方向準確率不等同扣除交易成本後的策略報酬。
- 請勿把模型結果當作個人化投資建議。

## 舊版 Streamlit
舊的 `app.py` 保留作為參考，不會由新的 GitHub Actions 流程執行。


## 新增技術指標與學習資料
- MA5 / MA20 / MA60：短中期趨勢參考。
- RSI(14)：價格動能指標；70/30僅為常見觀察區間，不是買賣指令。
- MACD(12,26,9)：趨勢動能與訊號線交叉。
- Bollinger Bands(20,2)：以均線與標準差觀察波動區間。
- ATR(14)：平均真實波幅，用來理解波動風險，不判斷方向。
- Volume ratio(20)：當日成交量相對20日均量。
- 以上技術特徵在訓練資料中至少落後一個台灣交易日；最新預測列會用最近可得行情重新計算。

## 研究參考入口
- TWSE：<https://www.twse.com.tw/>
- TAIFEX：<https://www.taifex.com.tw/>
- FRED：<https://fred.stlouisfed.org/>
- CBOE VIX：<https://www.cboe.com/tradable_products/vix/>
- 元大0050：<https://www.yuantaetfs.com/product/detail/0050/Basic_information>
- Yahoo Finance 0050：<https://finance.yahoo.com/quote/0050.TW/>

這些連結有助於查核資料與學習；未標示為已接入者，不會被宣稱已作為模型輸入。
