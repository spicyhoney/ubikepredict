# YouBike 缺車站點的持續風險預測與派車規劃

針對「目前已無車可借」的站點，預測 30 分鐘後是否仍缺車，再以供車庫存、道路時間和運補車載量安排取送。

- [預警系統](https://main.d2wv8lmkdla1vb.amplifyapp.com/)
- [派車系統](https://main.d2wv8lmkdla1vb.amplifyapp.com/dispatch/)
- [操作影片（1:59）](https://youtu.be/Mq1oi1dxpl4)

## 功能與架構

預警頁呈現最多 10 個優先站點、LightGBM 風險、SHAP 原因，以及 30 分鐘後的歷史答案。派車頁可選 1–5 台運補車、候選站數上限、載量、作業時間與補車／保留量，並可單獨查看各車路線。

預設先套警示門檻，再取 Top 10／20／30 候選；不為湊滿名額加入低風險站。風險優先 Greedy 為目標站選擇可行車輛及供車點，各車行車加作業時間不超過 30 分鐘。全車隊共用供車庫存，避免重複分配；同車可一次取車後連續送至多站。

| 元件 | 用途 |
| --- | --- |
| Amplify、React／Vinext | 網頁與瀏覽器端派車演算法 |
| API Gateway、Lambda | LightGBM／SHAP 推論；歷史揭曉獨立處理 |
| 私有 S3 | 模型、推論輸入與揭曉資料分開保存 |
| Bedrock | 將既有模型原因整理成摘要，不決定風險或路線 |
| Leaflet／OpenStreetMap、OSRM | 地圖、道路路徑與行車時間估計 |

## 公開版本的範圍

本版提供產品原始碼、合成資料的演算法測試、AWS 範本及 Kiro 規格。**不附主辦資料、站點車數快照、模型權重、特徵快取、道路矩陣、私有部署設定或金鑰。** 下載後可建置前端並執行公開測試；完整歷史回放與派車 Demo 需另備經授權的資產，不能只靠這份原始碼離線重現。

資產路徑與驗證方式見 [私有資產與資料界線](docs/private-assets.md)。舊研究報表、錄影、簡報、機器專用交接，以及已不使用的單車逐站操作元件已從新版樹狀目錄排除；本機原檔保留。

這次採一般提交更新，沒有改寫既有 Git 歷史；舊提交中原有的資料與模型仍可能被存取。這份原始碼整理不代表已完成歷史資料清除。

## 開發與檢查

需求：64-bit Python 3.12、Node.js 22.13 以上、pnpm 11.19.0。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-aws.txt
.\.venv\Scripts\python.exe scripts/check-public-source.py
.\.venv\Scripts\python.exe scripts/test-public-source.py
cd frontend
pnpm install --frozen-lockfile
pnpm test
pnpm run lint
pnpm exec tsc --noEmit
$env:NEXT_PUBLIC_API_BASE_URL = "https://your-api.example/prod"
pnpm run build
```

`pnpm dev` 啟動前端；將 `frontend/.env.example` 複製為 `.env.local` 並填入你有權使用的 API 位址。`example` 位址只用於建置驗證，不能提供預測。完整後端與供車功能的啟動步驟見資產文件；不要將機密放入 `NEXT_PUBLIC_*`。

公開 CI 驗證原始碼界線、合成資料測試、前端型別／lint／建置、SAM 及容器建置；不持有競賽資料或 AWS 部署憑證，也不自動部署。

## 模擬限制

展示以 6/29 09:00 的車數為起點；庫存隨模擬取送更新，不演化民眾借還。每次送站預設另加 3 分鐘取裝卸作業，與道路行車時間分開計算。道路估時不含即時交通、車型限制或車庫往返。兩個半小時快照不能證明中間從未恢復；歷史命中率也不等於實際調度的因果改善。

[模型卡](docs/model-card.md) · [API 契約](docs/api-contract.md) · [目前狀態](docs/STATE.md)
