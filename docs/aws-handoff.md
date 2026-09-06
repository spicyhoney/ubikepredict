# AWS 現場交接：只剩憑證、部署與雲端驗收

這份文件是比賽現場的操作手冊。先讀 [專案完整交接](project-handoff.md)，再按本頁執行。

## 先說清楚目前狀態

已完成並可在沒有 AWS 憑證的情況下測試：

- 靜態 Vinext 前端與 Amplify manual deployment 封裝流程。
- API Gateway HTTP API、兩個 Linux/amd64 Lambda 容器、兩個私有 S3 bucket、最小權限 IAM、CORS、CloudWatch logs 與 API throttle 的 SAM/CloudFormation 模板。
- Prediction Lambda 的 S3 runtime 資產下載與大小／SHA256 核對；Reveal Lambda 另行讀取 truth。
- API Gateway payload format 2.0 handlers，包含 `health`、`options`、`predict`、`explain`、`reveal`。
- LightGBM `pred_contrib=True` 解釋、完整 68 特徵的中文名稱、受控 Bedrock Converse 摘要、輸出驗證與固定範本 fallback。
- 前端 SHAP／營運摘要面板、地圖標記縮小與密集路線編號視覺避讓。
- 部署、前端打包、雲端驗收、安全清理腳本，以及 GitHub Actions CI。

尚未完成，因為目前沒有比賽 AWS 憑證：

- 未在真實 AWS 帳號建立 S3、ECR、Lambda、API Gateway、Amplify 或 CloudWatch 資源。
- 未在實際 Region 確認 Bedrock model／inference profile 存取，也未真正呼叫 Bedrock。
- 未從公開 Amplify HTTPS 網址完成端到端演示、CORS、權限與 cloud/local parity 驗收。

所以報告時可說「AWS-ready 程式與部署骨架已完成」，但在本頁的雲端驗收通過前，不可說「AWS／Bedrock 已部署成功」。

## 唯一主路徑

~~~text
Amplify Hosting（manual static deployment）
  ↓ NEXT_PUBLIC_API_BASE_URL
API Gateway HTTP API
  ├─ health / options / predict / explain → Prediction Lambda 容器
  │                                      ├─ private runtime S3
  │                                      ├─ LightGBM + SHAP
  │                                      └─ Bedrock Runtime（失敗就回固定範本）
  └─ reveal                            → Reveal Lambda 容器
                                         └─ private truth S3
~~~

不需要 SageMaker：模型小、演示流量低，Lambda 容器已足夠。AWS 只改變儲存、權限、服務位置與公開網址，不重訓模型，不改 68 欄順序、Platt 參數或門檻。

## 現場開始前準備

1. Clone Repository，讀完交接文件。
2. 安裝 Git、64-bit Python 3.12、Node.js 22.13 以上、AWS CLI v2、AWS SAM CLI 與 Docker Desktop（Linux container mode）。
3. 用 IAM Identity Center／SSO 或主辦方提供的短期憑證；不可把 Access Key 寫入 Git、`.env` 或網頁。
4. 選定 Region，並確認所選 Bedrock model 或 inference profile 在該帳號／Region 可用。
5. 先在本機執行全套測試：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\setup-frontend.ps1
.\scripts\smoke-test.ps1
~~~

測試數量可隨實作增加，不要只數固定幾項；以當次全套實跑成功及六月 parity 為準。

登入後確認身分：

~~~powershell
aws configure sso
aws sts get-caller-identity
~~~

## 先預演，不建立資源

`-DryRun` 會核對必要檔案、參數與 SAM 模板，不呼叫 AWS 建立資源：

~~~powershell
.\scripts\deploy-aws.ps1 -Region "ap-northeast-1" -DryRun
~~~

## 正式部署（有 AWS 憑證後）

以 Bedrock 啟用為例；`BedrockModelId` 可以是 direct model ID 或 inference-profile ID。若是 direct model，資源清單放該 foundation-model ARN；若是 cross-Region inference profile，必須同時放 profile ARN 與它可能路由到的 foundation-model ARNs，不能只給 profile ARN。可先用 `aws bedrock get-inference-profile` 讀取 `models[].modelArn`；Global profile 還要依 [AWS 官方 Global cross-Region IAM 說明](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html) 加入所需 regional／global FM ARN，並確認主辦帳號的 SCP 沒有阻擋目的 Region。

~~~powershell
$BedrockResourceArns = @(
  "arn:aws:bedrock:SOURCE_REGION:ACCOUNT:inference-profile/PROFILE_ID"
  "arn:aws:bedrock:DESTINATION_REGION::foundation-model/MODEL_ID"
)

.\scripts\deploy-aws.ps1 `
  -StackName "ubikepredict-demo" `
  -Region "ap-northeast-1" `
  -Profile "你的-aws-profile" `
  -EnableBedrock `
  -BedrockModelId "現場可用的-model-或-profile-id" `
  -BedrockModelResourceArns $BedrockResourceArns
~~~

若 profile 的 `models[]` 有多個 ARN，全部加入陣列。腳本會把這份明確清單寫入 Prediction role；不要為了省事改成 `Resource: "*"`。現場若無法確認 profile 的完整 IAM 範圍，先選可直接呼叫的單區模型，或先關閉 Bedrock 使用 template fallback。

腳本會依序：

1. 驗證 SAM 模板。
2. 透過 Docker/SAM 建置兩個 Lambda 容器映像並部署 CloudFormation stack。
3. 把凍結 runtime 資產與 reveal truth 上傳到兩個不同的私有 S3 bucket。
4. 以雲端 API URL 建置 Vinext 靜態網站，擋住任何 localhost 回退。
5. 使用 Amplify manual deployment 上傳前端 ZIP。
6. 執行雲端 health、predict、SHAP／摘要、reveal、CORS 與固定案例驗收；若使用 `-EnableBedrock`，摘要不是實際 `amazon_bedrock` 就會讓部署失敗。

上述是部署腳本自動驗收範圍。Prediction role 對 truth 的 `AccessDenied`、實際 throttle、另一台裝置開啟，以及 cloud/local parity（需先啟動本機 API 並另傳 `-LocalBaseUrl`）仍要依本頁「雲端必做驗收」逐項完成；不能只看到 stack 成功就宣稱全部驗收完成。

若一開始尚無 Bedrock 模型權限，可先不加 `-EnableBedrock` 完成其餘架構，此時 `operational_summary.provider` 會明確是 `template`，不可對外說成 Bedrock 產生。取得權限後重跑部署並完成 Bedrock 實際呼叫驗收。

## 腳本各自用途

~~~powershell
# 只重建可部署的靜態前端 ZIP（必須是 HTTPS API）
.\scripts\package-frontend.ps1 -ApiBaseUrl "https://API_ID.execute-api.REGION.amazonaws.com/prod"

# 對已部署 API 執行雲端驗收；加 LocalBaseUrl 會比較 cloud/local 機率
# 使用 LocalBaseUrl 前，另一個 PowerShell 視窗必須先執行 .\scripts\start-local-app.ps1
.\scripts\verify-cloud.ps1 `
  -StackName "ubikepredict-demo" `
  -Region "ap-northeast-1" `
  -Profile "你的-aws-profile" `
  -Origin "https://main.AMPLIFY_DOMAIN" `
  -LocalBaseUrl "http://127.0.0.1:8000"

# 要求驗收時必須真正由 Bedrock 回應；若降級為 template 便立即失敗
.\scripts\verify-cloud.ps1 -BaseUrl "https://API_URL/prod" -Origin "https://main.AMPLIFY_DOMAIN" -RequireBedrock

# 先只預覽將清理的 stack/buckets，不會刪除
.\scripts\cleanup-aws.ps1 -StackName "ubikepredict-demo" -Region "ap-northeast-1" -Profile "你的-aws-profile"

# 確認後才刪除這個 stack 擁有的版本化 bucket 內容與資源
.\scripts\cleanup-aws.ps1 -StackName "ubikepredict-demo" -Region "ap-northeast-1" -Profile "你的-aws-profile" -Execute -DeleteBucketContents
~~~

`package-frontend.ps1` 在 Windows 會明確呼叫 `vinext.CMD`，並只對 Vinext 已完成靜態匯出後的已知 Windows shutdown assertion 作受限容錯；其他建置錯誤仍會停止。產物必須含 `frontend/dist/client/index.html`，而且不可含 `localhost:8000`。

## S3 與 truth 權限邊界

Runtime bucket 包含：

~~~text
releases/frozen-v1/model/lgbm_full.txt
releases/frozen-v1/config/final_policy_freeze_before_may.json
releases/frozen-v1/config/protocol_frozen_before_june.json
replays/2026-06/input/dynamic_red_empty_2026_06_input.parquet
stations/dim_station.csv
~~~

Truth bucket 只包含：

~~~text
replays/2026-06/truth/june_all_eligible_decisions.parquet
~~~

兩個 bucket 都會啟用 Block Public Access、SSE-S3 與 Versioning。S3 loader 只接受程式中寫死的 allow-list keys，下載到 Lambda `/tmp/ubikepredict` 後必須通過檔案大小與 SHA256 才會被載入。

Prediction Lambda role 只能讀 runtime bucket 指定 objects，沒有 truth bucket 權限；Reveal Lambda role 只能讀指定 truth object，不初始化 LightGBM、不呼叫 Bedrock。

但 `POST /api/reveal` 是為了比賽回放而公開的 API 路由，使用者按「揭曉」後可取得所選站點的 30 分鐘後結果。安全主張應是「預測過程與 Prediction role 無法讀答案」，不是「答案從外部完全取得不到」。

## SHAP 與 Bedrock 的受控邊界

SHAP 不需重訓：LightGBM Booster 對同一筆 68 特徵使用 `pred_contrib=True`，前 68 欄為特徵對 raw score 的貢獻，最後一欄是 base value。測試會確認全部貢獻加總等於 LightGBM raw score。貢獻是原始分數單位，不能說成機率增加幾個百分點。

中文特徵名、顯示值、貢獻方向與排序均由伺服器程式決定。送給 Bedrock 的只有 allow-listed station facts 與最多 8 個已驗證 SHAP factors；Bedrock 只能從提供的 feature IDs 選 1～3 個並指定受限主題。最終中文文字由伺服器用已驗證的站點事實重新組裝，因此 Bedrock 不能改機率、門檻、排名、路線或真值，也不能發明天氣、活動或因果關係。

Bedrock 關閉、沒有 model ID、逾時、呼叫失敗或 JSON 不合規時，核心預測不會失敗；API 改回 `provider: "template"` 與 `fallback_reason`。只有實際成功呼叫且通過驗證時才回 `provider: "amazon_bedrock"`。

## API Gateway、CORS 與 throttle

路由：

~~~text
GET  /api/health
GET  /api/options
POST /api/predict
POST /api/explain
POST /api/reveal
~~~

POST 必須使用 `Content-Type: application/json`（可帶 charset），JSON body 不得超過 64KB。API Gateway 的預設 throttle 為 3 requests/second、burst 10；較慢的 `POST /api/explain` 另設 1 request/second、burst 2。數值是 CloudFormation 參數，現場若要改必須有理由並重新驗收。

CORS 只允許實際 Amplify HTTPS origin、開發用 `http://localhost:3000`，以及可選的精確 HTTPS override。前端不得持有 AWS credentials，不直接讀 S3，也不直接呼叫 Bedrock。

## 本機仍可獨立操作

上雲程式沒有移除本機模式。Windows PowerShell 在 Repository 根目錄執行：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start-local-app.ps1
~~~

開啟 `http://localhost:3000`。本機無 AWS 憑證時，SHAP 仍為真實 LightGBM 貢獻，營運摘要會明確標示為「本機備援／模型原因摘要」。

## GitHub Actions CI

`.github/workflows/ci.yml` 會在推送到 `main` 與所有 pull request 執行：

- 凍結模型 smoke test 與全套 Python/API/Lambda/S3/SHAP/Bedrock 測試。
- 前端 lint 與指向非 localhost HTTPS API 的靜態建置，並掃描產物是否殘留 localhost API。
- SAM/CloudFormation lint validation。
- Linux/amd64 Lambda 容器 build（不 push）。

CI 通過只能證明程式、模板與容器可建置，不能取代實際 AWS 部署與 Bedrock 存取驗收。

## 雲端必做驗收

1. 本機全套測試通過。
2. 公開 `/api/health` 與 `/api/options` 正常。
3. 預設三重案例為 23 個當下缺車、21 個可評分、12 個過平衡門檻、最終 Top-10。
4. Reveal 預設案例為 7/10，重複 station ID 不會灌高結果。
5. Cloud 與本機同站機率誤差不超過 `2e-7`。
6. Prediction role 讀 truth object 必須 `AccessDenied`；Reveal role 可讀指定 truth object。
7. SHAP 貢獻加總等於 raw score，且不含答案欄。
8. Bedrock 啟用時，`/api/explain` 實際回 `provider: "amazon_bedrock"`；關閉或故意讓其失敗時仍有 `template` fallback。
9. Amplify 無痕視窗可完成 predict、點站 explain 與 reveal，全程不依賴 localhost。
10. 非白名單 Origin 被擋，真實 Amplify origin 可用；請求不會在網頁暴露 AWS 金鑰、S3 truth URL 或答案欄。

## Demo 現場檢查

1. 在 Amplify 公開網址執行預測。
2. 點選一個可評分站，看 SHAP 推升／降低因素。
3. 如 `provider` 真為 `amazon_bedrock`，再說 Bedrock 已把受控因素整理成營運摘要；若為 `template`，就誠實展示 fallback。
4. 揭曉30分鐘後結果，說明這是公開 reveal 回放路由，不是 Prediction Lambda 偷看答案。
5. 快速展示 CloudFormation、兩個 Lambda roles、兩個私有 buckets 與 Bedrock 呼叫記錄。

## Definition of Done

只有同時符合下列條件，才可在簡報寫「AWS／Bedrock 已完成」：

- Amplify HTTPS 前端可從另一台裝置操作，不依賴 localhost。
- 兩個 Lambda 確實從各自私有 S3 讀取對應資產，Prediction role 不能讀 truth。
- SHAP 來自同一筆 LightGBM 推論，而且 sum check 通過。
- Bedrock 在實際帳號／Region 成功回應並通過 allow-list 驗證，失敗時 fallback 仍可用。
- Cloud/local parity、CORS、throttle、reveal 與權限邊界均驗收完成。
- 沒有使用 6 月重訓、重校正或重選門檻。
