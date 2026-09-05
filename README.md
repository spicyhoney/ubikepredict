# YouBike 30 分鐘缺車風險：可攜推論包

這個 Repository 已包含在另一台 Windows 筆電重現 Demo 所需的凍結模型、六月特徵、站點座標、門檻、機率校正設定及核對資料。正常 Demo 不需要 1～6 月原始 CSV，也不需要重新訓練。

## 第一次使用（Windows PowerShell）

先安裝 64-bit Python 3.12，Clone 後在專案根目錄執行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\smoke-test.ps1
.\scripts\run-demo.ps1
```

Smoke test 會驗證模型與資料檔雜湊、9 筆門檻兩側 golden cases、未知站點，以及六月 27,962 筆有效決策的完整重算。成功後才建議開始展示。

## 查詢指定時間、行政區與 Top-N

```powershell
.\scripts\run-demo.ps1 -Datetime "2026-06-23 19:30" -District "板橋區" -Policy balanced -Top 10
```

政策可選：

- `balanced`：校正機率至少 56.681205%，適合展示較完整的巡補候選。
- `strict`：校正機率至少 65.177727%，警示較少、要求較高。
- `all`：不先套門檻，直接列出該篩選範圍風險最高的站點。

也可直接呼叫 Python，輸出 JSON 給前端讀取：

```powershell
.\.venv\Scripts\python.exe .\backend\inference.py --datetime "2026-06-23 19:30" --district "板橋區" --policy all --top 10 --format json --output .\demo-output.json
```

## 資料與模型分層

```text
model/lgbm_full.txt
  凍結 LightGBM；只做推論，不在 Demo 重新訓練。
config/final_policy_freeze_before_may.json
  68 欄特徵順序、Platt 參數、平衡與嚴格門檻。
config/protocol_frozen_before_june.json
  六月開封前凍結的類別 schema 與實驗協議。
data/source/dynamic_red_empty_2026_06.parquet
  六月封存特徵快取；含 92,882 筆候選快照。
data/reference/june_all_eligible_decisions.parquet
  27,962 筆一次性六月核對結果，只供測試與揭曉。
data/stations/dim_station.csv
  站名、行政區、經緯度與容量範圍。
```

`MANIFEST.json` 記錄每個必要檔案的大小、SHA256 與用途；測試會在啟動前逐一核對，避免搬電腦時遺漏或拿錯版本。

## 推論公式與防洩漏

流程固定為：六月特徵 → 依 freeze 的 68 欄與順序建矩陣 → LightGBM `raw_score=True` → Platt sigmoid → 套用凍結門檻。

六月 source 為了離線稽核仍保留 `y_same_30` 與 `future_30_*`。推論程式採「特徵白名單」，只選 freeze JSON 列出的 68 欄；答案、未來資訊及 reference 絕不會傳入模型。部署前端時也不要把整個 `data/` 當公開靜態目錄。

## 實驗時間界線

- 2026 年 1～3 月：模型訓練。
- 2026 年 4 月：機率校正、門檻選擇及 gate。
- 2026 年 5 月：凍結後的探索性確認，未用於重訓或選門檻。
- 2026 年 6 月：一次性最終測試與 Demo 回放；沒有用來調參。

## GitHub 應放與不應放的內容

本專案內的模型、兩份 JSON、兩份 Parquet 與站點 CSV 都應提交，總量很小。不要提交 `.venv`、`node_modules`、AWS 金鑰、真正的 `.env`、1 GB 原始 CSV、`tmp` 或舊實驗輸出。比賽前請在實際筆電全新 Clone 一次並跑 smoke test，另將 Repository ZIP 備份到 USB。

## AWS 串接邊界

未串 AWS 時，本包已可離線完整推論。之後可將 `FrozenYouBikeModel.predict_frame()` 包進 Lambda／SageMaker API；AWS 只負責部署、權限、儲存與 API，不應改動 68 欄順序、校正參數或門檻。

