# 0050 AI Forecast V1

## 目標設定
- 開盤前預測 0050 當日收盤相對前一交易日收盤方向
- 五年歷史資料
- 雲端網頁儀表板
- 50 項候選因子登錄表
- Walk-forward 樣本外回測

## 本版本的誠實範圍
這是可部署的研究起始專案，不是已完成的 50 因子生產模型。
首版可由 Yahoo Finance 下載一批可取得的市場價格序列；官方 TWSE/TAIFEX 歷史資料、新聞情緒、法人現貨買賣超、產業動態指數及部分海外資料仍需接入。
因子登錄表會標示可計算、待接 API 或待建置，不會把缺資料的項目假裝已納入模型。
Yahoo Finance 是非官方資料來源，正式部署應以 TWSE、TAIFEX、可信行情供應商作交叉驗證。

## 本機啟動
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 雲端部署
可部署到 Streamlit Community Cloud 或其他支援 Python 的主機。將此專案推送到 GitHub，建立 Streamlit app，指定 `app.py`。
正式長期服務建議使用獨立排程器與資料庫；不要把 Streamlit 單一程序當成可靠的每日資料倉儲。

## 重要研究限制
1. 預測標籤是下一交易日 close > 當前 close；最後一列沒有標籤。
2. 全球市場資料使用前一交易日觀測值；正式版需以交易所時區與交易日曆精確對齊。
3. 台股收盤後公布的法人資料只能作為下一交易日特徵，不能用於當日開盤前預測。
4. 回測以 expanding-window walk-forward，不隨機切分時間序列。
5. 本版報告為方向準確率，不等同可交易策略報酬；尚未納入滑價、手續費、稅負與交易限制。
6. 不保證 90% 正確率。若樣本外測試未達標，應如實顯示。

## 下一階段
- 官方 TWSE/TAIFEX API 接入及快取
- 歷史新聞資料與發布時間戳記、新聞去重及情緒分類
- 動態 0050 成分股權重與成分股變更歷史
- FRED 利率資料及韓國股市資料
- 特徵重要性、Permutation importance、SHAP
- 模型版本管理、漂移偵測、Champion/Challenger 部署流程
- 雲端資料庫、排程、監控與通知
