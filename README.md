# 0050 AI Forecast

以 GitHub Actions 執行每日資料更新、樣本外回測與下一交易日方向研究；GitHub Pages 提供靜態預測儀表板，不需要 Streamlit 常駐服務。

## 架構
- `data.py`：下載研究用市場價格資料（Yahoo Finance，非官方來源）。
- `features.py`：產生歷史特徵與標籤。
- `model.py`：Logistic Regression 基準模型與 expanding-window Walk-forward 回測。
- `run_daily.py`：由 GitHub Actions 呼叫，輸出 `site/data/latest.json`。
- `site/index.html`：不依賴 Streamlit 的靜態儀表板。
- `.github/workflows/daily.yml`：排程執行並部署 GitHub Pages。

## 啟用方式
1. 在 GitHub Repository 的 **Settings → Actions → General → Workflow permissions** 確認工作流程允許必要的讀寫權限。
2. 在 **Settings → Pages** 將 **Build and deployment → Source** 設為 **GitHub Actions**。
3. 前往 **Actions → 0050 AI Daily Forecast → Run workflow**，手動執行第一次流程。
4. 成功後從 **Settings → Pages** 查看網站網址。排程為星期一至星期五 UTC 00:30（台灣時間 08:30），但 GitHub 排程可能延遲，且不是交易日曆。
5. 若工作流程失敗，開啟 **Actions** 查看失敗步驟與錯誤紀錄。

## 研究限制
- Yahoo Finance 是非官方資料來源，行情完整性需自行交叉驗證。
- 目前使用 Logistic Regression 基準分類器，不保證準確率或獲利。
- 跨市場時區與交易日曆尚未完整建模；盤前可用性需要進一步驗證。
- 回測方向準確率不等同扣除交易成本後的策略報酬。
- 請勿把模型結果當作個人化投資建議。

## 舊版 Streamlit
舊的 `app.py` 保留作為參考，不會由新的 GitHub Actions 流程執行。
