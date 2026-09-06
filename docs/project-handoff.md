# 專案完整交接：新筆電／新對話先讀

這份文件是此專案的「記憶」。新筆電、新隊友或新的 AI 對話在改任何程式前，應先讀完本文件、[模型卡](model-card.md)、[API 契約](api-contract.md)及 [AWS 現場交接](aws-handoff.md)。

## 一句話版本

官方地圖已能告訴管理者當下失衡；本作品使用已凍結的 LightGBM，以「現在已0車，預測30分鐘後是否仍0車」為主模式，並以「現在已0空位，預測30分鐘後是否仍0空位」為輔助模式，讓短暫紅燈與值得持續關注的紅燈分流。

## 最終預測問題

- `mode=empty`（主）：輸入是當下 `available_bikes == 0` 的站點快照；30分鐘後仍0車時 y=1，已恢復有車時 y=0。
- `mode=full_dock`（輔）：輸入是當下 `available_docks == 0` 的站點快照；30分鐘後仍0空位時 y=1，已恢復空位時 y=0。
- 輸出：各自經 Platt 校正的「30分鐘後仍失衡」機率。
- 使用方式：先套凍結門檻，再產生有限行動清單；下一張快照重新判斷。
- 評分單位：每一個決策快照／站點各算一次，不使用「一整段 Episode 只中一次就算成功」。

這不是預測30分鐘後的精確車數／空位數，也不是預測一個原本正常的站會不會突然失衡，更不是實際派車最佳化。

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
| 6月 | 凍結結果核對及歷史回放 | 不可用來訓練、校正或選門檻 |

六月沒有進入任一模型的訓練、機率校正或門檻選擇。但六月快取已在專案過程中存在，特別是 `full_dock` 的數字只能表述為「凍結後回溯核對」，不應說成從未看過的前瞻盲測。

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

上圖是 `empty` 主模式的數值。`full_dock` 使用自己的模型，Platt 為 `sigmoid(0.8807565569877625 × raw score - 0.15054720640182495)`，平衡／嚴格門檻分別為 42.059604% 與 48.107435%。其餘特徵白名單、Top-K 與路線流程相同。

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
| model/lgbm_full_dock.txt | 滿柱輔助模式的114棵樹 | Demo 不重訓 |
| config/final_policy_freeze_before_may.json | 68欄順序、Platt參數、兩門檻 | 不得改值 |
| config/final_policy_freeze_full_dock_before_may.json | 滿柱的68欄順序、Platt參數、兩門檻 | 不得改值 |
| config/protocol_frozen_before_june.json | 類別 schema 與凍結協議 | 未知類別走 missing |
| config/protocol_full_dock_frozen_before_june.json | 滿柱類別 schema 與凍結協議 | 未知類別走 missing |
| data/source/dynamic_red_empty_2026_06_input.parquet | API 的 truth-free 模型輸入 | 可供 predict 讀取 |
| data/source/dynamic_red_full_2026_06_input.parquet | 滿柱 API 的 truth-free 模型輸入 | 可供 predict 讀取 |
| data/reference/june_all_eligible_decisions.parquet | 六月核對與揭曉答案 | predict 不得讀取 |
| data/reference/june_full_dock_all_eligible_decisions.parquet | 滿柱六月核對與揭曉 | predict 不得讀取 |
| data/source/dynamic_red_empty_2026_06.parquet | 離線 smoke/parity 稽核 | 含未來欄，只能私有 |
| data/stations/dim_station.csv | 站名、行政區、座標、容量 | 供顯示及路線 |
| MANIFEST.json | 核心推論資產的大小與 SHA256 | 不是整站供應鏈清單 |
| backend/lambda_handler.py | API Gateway v2 的 Prediction／Reveal handlers | 兩個 Lambda 分開權限 |
| backend/s3_assets.py | S3 allow-list 下載與大小／SHA256 核對 | 只在 AWS 模式需要 boto3 |
| backend/explanations.py | LightGBM SHAP 貢獻與中文特徵顯示 | 加總必須重建 raw score |
| backend/bedrock_summary.py | 受控 Bedrock 摘要與固定範本 fallback | 不得改機率、門檻或排名 |
| infra/template.yaml | S3、兩 Lambda 容器、API Gateway、IAM、Amplify | 已寫好，尚未在真實 AWS 部署 |

## 六月凍結結果

| 決策方式 | Precision | Recall | 解讀 |
|---|---:|---:|---|
| 所有當下紅燈都猜會持續 | 42.63% | 100% | 不懂資料的營運基準 |
| LightGBM 平衡門檻 | 65.84% | 21.26% | 主 Demo 政策 |
| LightGBM 嚴格門檻 | 68.56% | 7.00% | 警示少、較保守 |

以上是六月27,962筆有效決策的逐站逐快照結果。畫面中的單一案例（例如7/10或8/8）只是歷史回放，不得拿來冒充六月整體 Precision。

滿柱 `full_dock` 輔助模式的6月 3,776筆回溯結果：平衡版 Precision 47.71%、Recall 17.20%；嚴格證據版 Precision 56.79%、Recall 10.14%。它是輔助證據，不取代上表缺車主結果，也不能將單一案例 1/2 或 2/2 當成整體準確度。

Top-K 是營運清單上限，不是模型評分方式。先過機率門檻，再受K限制；沒有足夠高風險站時不能硬湊。圖上還要區分：

- 行動清單。
- 已過門檻但清單額滿。
- 30分鐘風險未達門檻。
- 歷史長度不足。
- 揭曉後仍缺車／已恢復。

## 已完成的系統

- 缺車主模式與滿柱輔助模式的真實 LightGBM、Platt 與凍結門檻推論。
- truth-free predict input 與延後讀取的 reveal truth。
- 全候選站、資料不足狀態、門檻與有限清單。
- 以實際經緯度畫站點並產生巡補順序示意。
- 歷史結果揭曉及逐站命中數。
- 本機 API、互動前端、WebMCP工具、CORS白名單。
- LightGBM `pred_contrib=True` 的 SHAP 原始分數貢獻、完整特徵中文名、數值與推升／降低風險方向。
- 受控 Bedrock Converse client、只能選 allow-listed SHAP feature IDs 的輸出驗證、伺服器組文及 template fallback。
- 前端站點解釋／營運摘要面板、縮小地圖標記與密集編號視覺避讓。
- S3 資產下載與雜湊核對、Prediction／Reveal Lambda handlers、Linux 容器、API Gateway／IAM／雙 S3／Amplify 的 SAM 模板。
- 部署、靜態前端封裝、雲端驗收、安全清理腳本與 GitHub Actions CI。
- Windows 一鍵安裝／啟動、全套 API／模型／Lambda／S3／SHAP／Bedrock 測試，及缺車27,962筆與滿柱3,776筆完整 parity。測試數量以最終實跑為準，不再寫死項目數。

## 尚未完成：只剩真實 AWS 憑證才能做的事

- 尚未取得比賽短期 AWS 憑證。
- 尚未確定實際 Region、該帳號可用的 Bedrock model／inference profile ID，以及 profile 與目的模型所需的精確 ARN 清單。
- 尚未執行 `scripts/deploy-aws.ps1` 建立真實 AWS 資源、上傳資產及 Amplify manual static 前端。
- 尚未在真實帳號驗收 Bedrock、Prediction role 無 truth 權限、CORS、公開 HTTPS Demo 與 cloud/local parity。

## 前端上雲與視覺狀態

- Vinext 已設為靜態匯出，`scripts/package-frontend.ps1` 可產生 Amplify manual deployment ZIP。
- Production build 強制提供非 localhost 的 HTTPS `NEXT_PUBLIC_API_BASE_URL`，並掃描建置產物是否殘留 localhost API。
- Windows 封裝腳本明確使用 `vinext.CMD`，並對已知 shutdown assertion 採受限容錯。
- 地圖標記已縮小，密集行動站的路線編號會視覺避讓，點選站點可開啟 SHAP／營運摘要面板。
- 尚未完成的是真實 Amplify 上傳、公開網址與跨裝置驗收，不是前端程式本身。

SHAP 是原因來源；Bedrock只能整理已提供的 SHAP 與站點事實，不得自行發明原因、改風險分數、改門檻或決定清單。

## 防洩漏與不可更動原則

1. /api/predict 只能讀取所選 `mode` 的 truth-free input；不得讀任一 reference 或原始稽核 Parquet。
2. /api/reveal 才能讀對應 reference，並使用獨立 AWS function／IAM role。
3. 6月不可重訓、校正或選門檻。
4. 68欄順序、類別 schema、Platt參數與兩個門檻不可因搬 AWS 而改變。
5. Bedrock不得看到六月答案，不得以生成內容取代模型計算。
6. 不得把 data/ 直接公開成前端靜態目錄。
7. 路線只是示意，不能宣稱為車隊最佳化或實際派車指令。
8. Bedrock 僅能從 allow-listed SHAP feature IDs 選重點；只有真正呼叫成功時前端才能標示 Amazon Bedrock。
9. AWS 的 `/api/reveal` 是公開 Demo endpoint；應宣稱的是 Prediction role 無權讀 truth，不是 truth 從外部完全無法取得。

## 新筆電首先執行

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\setup-frontend.ps1
.\scripts\smoke-test.ps1
.\scripts\start-local-app.ps1
~~~

Smoke test 必須同時顯示缺車27,962筆與滿柱3,776筆 parity，且最大機率差不超過 `2e-7`，並完成當前全套 API／Lambda／S3／SHAP／Bedrock fallback 測試，之後才執行 AWS 部署。

## 給新 AI 對話的開場提示

~~~text
請先完整閱讀 README.md、docs/project-handoff.md、docs/aws-handoff.md、
docs/api-contract.md 與 docs/model-card.md，再檢查 git status 與 GitHub Actions。
這是已凍結的 YouBike 30分鐘持續失衡模型；empty缺車是主模式，full_dock滿柱是輔助模式。
不得重訓、不得使用6月訓練／校正／選門檻，也不得改任一模型的68欄順序、Platt參數或門檻。
AWS-ready 程式、SAM、容器、SHAP、
Bedrock client/fallback、靜態前端與部署腳本已完成，但尚未實際上雲。取得比賽憑證後，
依 docs/aws-handoff.md 確認Region與Bedrock權限，執行deploy及cloud verification。不要把AWS-ready誤寫成已部署。
~~~
