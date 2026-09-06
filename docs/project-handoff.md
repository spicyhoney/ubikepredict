# 專案完整交接：新筆電／新對話先讀

這份文件是此專案的「記憶」。新筆電、新隊友或新的 AI 對話在改任何程式前，應先讀完本文件、[模型卡](model-card.md)、[API 契約](api-contract.md)及 [AWS 現場交接](aws-handoff.md)。

## 一句話版本

官方地圖已能告訴管理者「現在是0車」；本作品使用已凍結的 LightGBM，在站點現在已是0車的條件下，預測同一站30分鐘後是否仍為0車，讓短暫紅燈與值得持續關注的紅燈分流。

## 最終預測問題

- 輸入單位：某站某個30分鐘快照，且當下 available_bikes == 0。
- 標籤：同一站30分鐘後仍為0車時 y=1；已恢復有車時 y=0。
- 輸出：經 Platt 校正的「30分鐘後仍缺車」機率。
- 使用方式：先套凍結門檻，再產生有限行動清單；下一張快照重新判斷。
- 評分單位：每一個決策快照／站點各算一次，不使用「一整段 Episode 只中一次就算成功」。

這不是預測30分鐘後的精確車數，也不是預測一個原本正常的站會不會突然缺車，更不是實際派車最佳化。

## 為何改成這個方向

探索資料後發現，約六成缺車事件在下一張30分鐘快照就恢復。若把所有紅燈都當成同等重要，管理者會被大量短暫事件分散注意力。因此模型要辨識的是剩下較可能延續的紅燈，而不是重做官方即時地圖。

完整探索工作簿及決策順序見 [analysis/README.md](../analysis/README.md)。早期曾做一般變動、總變動、失衡率、高價值核心站、逐快照新失衡預警、Top-K、雙模型與存活模型等實驗；最終以「當下已缺車，預測30分鐘後是否持續」作為可解釋且較接近營運決策的版本。

## 資料時間界線

| 時間 | 用途 | 可否再調整模型 |
|---|---|---|
| 1～3月 | LightGBM 訓練 | 已完成，不需在 Demo 重訓 |
| 4月1～14日 | Platt 機率校正 | 已凍結 |
| 4月15～20日 | 選擇平衡／嚴格門檻 | 已凍結 |
| 4月21～30日 | 最後 gate | 已凍結 |
| 5月 | 凍結後探索性確認 | 不重訓、不重選門檻 |
| 6月 | 一次性最終測試及歷史回放 | 絕對不可用來調參 |

現場 AWS 串接是「搬移既有推論」，不是重新訓練。若 AWS 與本機輸出不同，先找環境、欄位順序、類別 schema 或檔案版本問題，不可用重訓掩蓋。

## 凍結模型怎麼使用

~~~text
truth-free 六月輸入
  → 補出3個已定義的持續時間衍生欄
  → 嚴格依 freeze JSON 取68欄及固定順序
  → 套固定 categorical schema
  → LightGBM 輸出 raw score
  → sigmoid(1.1206777095794678 × raw score + 0.11242114752531052)
  → 校正後風險機率
  → 平衡門檻56.681205%／嚴格門檻65.177727%
  → 最多K站的行動清單
  → 風險優先、相鄰距離串接的路線示意
~~~

68欄不是由前端送入，也不是現場人工重建。它們已存在 truth-free Parquet 中；backend/inference.py 只按照 config/final_policy_freeze_before_may.json 的白名單取用。特徵大致包含：

- 當下車／柱數、容量與比例。
- 前30、60、90分鐘的車柱數、比例、差分與加速度。
- 前90分鐘統計量。
- 前1天、前7天同時段狀態。
- 尖峰、週末、時刻與星期週期。
- 站點、行政區、經緯度。
- 同方向／反方向紅燈歷史。
- 缺車已持續多久及疑似大量跳動標記。

模型真正使用的完整名稱與順序只能以 freeze JSON 為準。

## 核心資產各自用途

| 檔案 | 用途 | 注意 |
|---|---|---|
| model/lgbm_full.txt | 已訓練的242棵樹 | Demo 不重訓 |
| config/final_policy_freeze_before_may.json | 68欄順序、Platt參數、兩門檻 | 不得改值 |
| config/protocol_frozen_before_june.json | 類別 schema 與凍結協議 | 未知類別走 missing |
| data/source/dynamic_red_empty_2026_06_input.parquet | API 的 truth-free 模型輸入 | 可供 predict 讀取 |
| data/reference/june_all_eligible_decisions.parquet | 六月核對與揭曉答案 | predict 不得讀取 |
| data/source/dynamic_red_empty_2026_06.parquet | 離線 smoke/parity 稽核 | 含未來欄，只能私有 |
| data/stations/dim_station.csv | 站名、行政區、座標、容量 | 供顯示及路線 |
| MANIFEST.json | 核心推論資產的大小與 SHA256 | 不是整站供應鏈清單 |

## 六月凍結結果

| 決策方式 | Precision | Recall | 解讀 |
|---|---:|---:|---|
| 所有當下紅燈都猜會持續 | 42.63% | 100% | 不懂資料的營運基準 |
| LightGBM 平衡門檻 | 65.84% | 21.26% | 主 Demo 政策 |
| LightGBM 嚴格門檻 | 68.56% | 7.00% | 警示少、較保守 |

以上是六月27,962筆有效決策的逐站逐快照結果。畫面中的單一案例（例如7/10或8/8）只是歷史回放，不得拿來冒充六月整體 Precision。

Top-K 是營運清單上限，不是模型評分方式。先過機率門檻，再受K限制；沒有足夠高風險站時不能硬湊。圖上還要區分：

- 行動清單。
- 已過門檻但清單額滿。
- 30分鐘風險未達門檻。
- 歷史長度不足。
- 揭曉後仍缺車／已恢復。

## 已完成的系統

- 真實 LightGBM、Platt 與凍結門檻推論。
- truth-free predict input 與延後讀取的 reveal truth。
- 全候選站、資料不足狀態、門檻與有限清單。
- 以實際經緯度畫站點並產生巡補順序示意。
- 歷史結果揭曉及逐站命中數。
- 本機 API、互動前端、WebMCP工具、CORS白名單。
- Windows 一鍵安裝／啟動、13項 API/模型測試及27,962筆完整 parity。

## 尚未完成，不能假裝已完成

- SHAP 貢獻計算與中文特徵名稱映射。
- 前端「主要原因／AI營運摘要」面板。
- Amazon Bedrock 呼叫、prompt、JSON驗證、快取及失敗 fallback。
- S3下載器、Lambda handler／容器、API Gateway與IAM。
- AWS上的端到端驗收；前端雲端部署也是選配而非既成事實。

SHAP 是原因來源；Bedrock只能整理已提供的 SHAP 與站點事實，不得自行發明原因、改風險分數、改門檻或決定清單。

## 防洩漏與不可更動原則

1. /api/predict 只能讀 truth-free input；不得讀 reference 或原始稽核 Parquet。
2. /api/reveal 才能讀 reference，且最好使用獨立 AWS function／IAM role。
3. 6月不可重訓、校正或選門檻。
4. 68欄順序、類別 schema、Platt參數與兩個門檻不可因搬 AWS 而改變。
5. Bedrock不得看到六月答案，不得以生成內容取代模型計算。
6. 不得把 data/ 直接公開成前端靜態目錄。
7. 路線只是示意，不能宣稱為車隊最佳化或實際派車指令。

## 新筆電首先執行

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\setup-frontend.ps1
.\scripts\smoke-test.ps1
.\scripts\start-local-app.ps1
~~~

Smoke test 必須顯示27,962筆 parity 且最大機率差為0，之後才開始 AWS 修改。

## 給新 AI 對話的開場提示

~~~text
請先完整閱讀 README.md、docs/project-handoff.md、docs/aws-handoff.md、
docs/api-contract.md 與 docs/model-card.md，再檢查 git status。
這是已凍結的 YouBike 30分鐘持續缺車模型；不得重訓、不得使用6月調參，
也不得改68欄順序、Platt參數或門檻。請從 docs/aws-handoff.md 的未完成清單
繼續，完成一項就更新文件與測試，不要把規劃誤寫成已完成。
~~~
