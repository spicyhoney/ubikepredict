# AWS 現場交接：S3、Lambda、API Gateway、SHAP、Bedrock

這是比賽現場的實作清單，不是完成宣告。開始前先讀 [專案完整交接](project-handoff.md)。目前 Repository 已完成本機模型、API與前端；本文件中標示「待完成」的 AWS、SHAP及 Bedrock 程式仍需在有 AWS 帳號的環境實作與驗收。

## 建議只走這條主路

依目前約1.24MB模型、2.86MB truth-free輸入及低頻 Demo 流量，第一版採：

~~~text
本機前端（先求穩定，雲端託管為選配）
  ↓ NEXT_PUBLIC_API_BASE_URL
API Gateway HTTP API
  ├─ health / options / predict → Prediction Lambda 容器
  │                              ├─ S3 runtime bucket
  │                              ├─ LightGBM + SHAP
  │                              └─ Bedrock Runtime
  └─ reveal                    → Reveal Lambda 容器
                                 └─ S3 truth bucket
~~~

先不要同時做 SageMaker。只有 Lambda 冷啟動經實測不可接受、流量持續且高，或確實需要託管端點監控時，再評估 SageMaker Real-time Endpoint。

## 現在已完成與待完成

| 項目 | 狀態 |
|---|---|
| 凍結 LightGBM、68欄、Platt、兩門檻 | 已完成 |
| truth-free六月輸入、獨立reveal reference | 已完成 |
| 本機 /api/health、options、predict、reveal | 已完成 |
| 互動前端、Top-K、地圖、路線、揭曉 | 已完成 |
| S3下載器與雜湊核對 | 待完成 |
| Lambda handler與Linux容器 | 待完成 |
| API Gateway、IAM與正式CORS | 待完成 |
| LightGBM pred_contrib／SHAP原因 | 待完成 |
| Bedrock Runtime呼叫與fallback | 待完成 |
| 前端「AI營運摘要」面板 | 待完成 |

重要：backend/api.py 是本機 ThreadingHTTPServer，不是 Lambda handler；現有 UBIKE_*_PATH 只接受本機路徑，不能直接填 s3://。

## 0. 新筆電與帳號先確認

1. Clone Repository，完整閱讀交接文件。
2. 安裝 Git、64-bit Python 3.12、Node.js 22.13以上、AWS CLI v2及Docker Desktop。
3. 使用自己的 IAM Identity Center／SSO暫時憑證，不可交換或提交長期Access Key。
4. 選定一個有可用 Bedrock 模型的 Region，S3、ECR、Lambda、API Gateway盡量同Region。
5. 先在本機執行：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\setup-frontend.ps1
.\scripts\smoke-test.ps1
~~~

必須看到27,962筆 parity 且最大機率差為0，再碰 AWS。

登入 AWS 後核對身分：

~~~powershell
aws configure sso
aws sts get-caller-identity
~~~

## 1. S3 分層

建立兩個私有 bucket，名稱自行加 account及region以避免撞名：

~~~text
s3://ubike-demo-runtime-<account>-<region>/
  releases/frozen-v1/model/lgbm_full.txt
  releases/frozen-v1/config/final_policy_freeze_before_may.json
  releases/frozen-v1/config/protocol_frozen_before_june.json
  replays/2026-06/input/dynamic_red_empty_2026_06_input.parquet
  stations/dim_station.csv

s3://ubike-demo-truth-<account>-<region>/
  replays/2026-06/truth/june_all_eligible_decisions.parquet
~~~

要求：

- 兩個bucket都啟用Block Public Access與Versioning。
- 瀏覽器只呼叫API Gateway，不可直接讀S3資料。
- Prediction role完全不能讀truth bucket。
- Reveal role只需讀truth物件，不需要Bedrock權限。
- 不把含未來欄的 dynamic_red_empty_2026_06.parquet 給Prediction Lambda。
- release key使用 frozen-v1 或commit SHA，不依賴 latest。

上傳範例：

~~~powershell
$RuntimeBucket = "你的-runtime-bucket"
$TruthBucket = "你的-truth-bucket"
aws s3 cp model/lgbm_full.txt "s3://$RuntimeBucket/releases/frozen-v1/model/lgbm_full.txt"
aws s3 cp config/final_policy_freeze_before_may.json "s3://$RuntimeBucket/releases/frozen-v1/config/final_policy_freeze_before_may.json"
aws s3 cp config/protocol_frozen_before_june.json "s3://$RuntimeBucket/releases/frozen-v1/config/protocol_frozen_before_june.json"
aws s3 cp data/source/dynamic_red_empty_2026_06_input.parquet "s3://$RuntimeBucket/replays/2026-06/input/dynamic_red_empty_2026_06_input.parquet"
aws s3 cp data/stations/dim_station.csv "s3://$RuntimeBucket/stations/dim_station.csv"
aws s3 cp data/reference/june_all_eligible_decisions.parquet "s3://$TruthBucket/replays/2026-06/truth/june_all_eligible_decisions.parquet"
~~~

不要將帳號ID、金鑰或登入資訊寫入GitHub。

## 2. AWS 後端需要新增的檔案

建議新增並測試：

- backend/s3_assets.py：冷啟動時下載固定S3 objects到 /tmp/ubikepredict，核對大小與SHA256。
- backend/lambda_handler.py：解析API Gateway HTTP API payload format 2.0並呼叫現有DemoService方法。
- backend/explanations.py：計算LightGBM貢獻、中文原因映射、固定模板fallback。
- backend/bedrock_summary.py：Bedrock Runtime client、受控prompt、JSON schema驗證、timeout及cache。
- Dockerfile.aws：以AWS Lambda Python 3.12 base image安裝固定版科學套件。
- infra/：使用SAM、CDK或Terraform擇一，記錄S3、ECR、兩個Lambda、API Gateway、IAM及環境變數。
- tests/test_explanations.py、tests/test_lambda_handler.py：本機不連AWS也能測核心邏輯。

不要把 backend/api.py 原封不動上傳就宣稱是Lambda；必須新增真正handler，或明確採用並測試Lambda Web Adapter。對目前專案而言，直接handler較容易控制。

## 3. 冷啟動與檔案路徑

Prediction Lambda初始化階段：

1. 從runtime bucket下載模型、兩份config、truth-free input與station CSV到 /tmp/ubikepredict。
2. 下載後核對預期SHA256；錯誤就停止，不可帶錯版本繼續。
3. 將既有環境變數指向 /tmp 中的本機檔案。
4. 在handler外只初始化一次 FrozenYouBikeModel 與資料，warm invocation重用。

Reveal Lambda只下載reference，最好做成輕量服務，不初始化模型。

目前既有環境變數：

~~~text
UBIKE_MODEL_PATH
UBIKE_FREEZE_PATH
UBIKE_PROTOCOL_PATH
UBIKE_API_INPUT_PATH
UBIKE_REFERENCE_PATH
UBIKE_STATIONS_PATH
UBIKE_ALLOWED_ORIGINS
NEXT_PUBLIC_API_BASE_URL
~~~

建議新增，但目前尚未實作：

~~~text
AWS_REGION
UBIKE_RUNTIME_BUCKET
UBIKE_TRUTH_BUCKET
UBIKE_RELEASE_ID=frozen-v1
BEDROCK_ENABLED=true
BEDROCK_MODEL_ID=<現場已取得存取權的model或inference profile>
BEDROCK_TIMEOUT_SECONDS=<短於API Gateway總timeout>
~~~

Lambda容器是Linux且檔案系統唯讀，只有 /tmp 可寫。從Windows建置時明確使用 linux/amd64；若選ARM，LightGBM、NumPy、Pandas及PyArrow都要重新驗證。

## 4. SHAP／LightGBM貢獻怎麼做

不必重訓模型，也不一定要新增 shap 套件。LightGBM Booster可用 pred_contrib=True產生每欄貢獻：

~~~text
matrix = engine.model_matrix(frame)
contrib = engine.booster.predict(matrix, pred_contrib=True)
~~~

- 前68欄對應凍結特徵，最後一欄是expected value。
- 每列貢獻加總應等於該列LightGBM raw score；寫測試核對容許誤差。
- 以絕對值排序，但前端分開顯示「提高風險」與「降低風險」。
- SHAP值是raw-score貢獻，不能直接說成「增加幾個百分點機率」。
- 原始欄名先經固定中文映射／主題分組，不可交給Bedrock自行猜欄位含義。
- 缺值、站點ID、經緯度等特徵需要可理解的說法；無法安全翻譯就顯示較高層主題。

建議在 predict response 的行動站加入：

~~~json
{
  "top_factors": [
    {
      "feature": "red_duration_capped_6",
      "label": "缺車已持續一段時間",
      "direction": "increase",
      "contribution": 0.42
    }
  ]
}
~~~

這是確定性模型解釋，即使Bedrock失敗也要保留。

## 5. Bedrock只負責整理

建議新增 POST /api/explain，而不要讓每次predict都等待生成：

~~~json
{
  "decision_time": "2026-06-29T09:00:00+08:00",
  "station_id": 673
}
~~~

後端自行找出該站已計算的風險、持續時間、路線順位及top_factors，再把受控JSON送給Bedrock。Bedrock輸出固定schema：

~~~json
{
  "headline": "30分鐘後仍缺車風險偏高",
  "reasons": [
    "缺車已持續一段時間",
    "近期車輛變化提高持續風險"
  ],
  "action_note": "建議保留於目前巡補關注清單",
  "disclaimer": "此為決策輔助，不是實際派車指令"
}
~~~

Prompt必須要求：

- 只能使用輸入JSON中的事實。
- 不可創造天氣、活動、交通、因果關係或精確需求量。
- 不可改風險機率、門檻、排名、路線或真實結果。
- 找不到原因時明確說資料不足。
- 只回指定JSON，後端驗證後才送前端。

Bedrock失敗、逾時、429或輸出不合法時，立即使用固定模板整理SHAP；核心預測不可一起失敗。前端只有收到真實Bedrock provider標記時才顯示「Amazon Bedrock」，fallback必須標為「模型原因摘要」，不能假裝是Bedrock生成。

正式示範前必須由帳號管理者在實際Region確認所選模型可呼叫；第三方模型第一次使用可能還需要先完成模型存取或Marketplace流程。

## 6. 前端需要新增

點選行動站後，在地圖下方或右側加入「AI營運摘要」：

- 站名、風險、門檻、路線順位。
- SHAP主要推升因素與降低因素。
- Bedrock中文摘要或明確fallback。
- loading、timeout、失敗與重新整理狀態。
- 清楚註明「解釋不影響模型分數」。

API Gateway URL確定後，在 frontend/.env.local 設：

~~~text
NEXT_PUBLIC_API_BASE_URL=https://<api-id>.execute-api.<region>.amazonaws.com
~~~

這是前端build-time值；改URL後必須重新啟動或build。先讓本機前端穩定呼叫AWS API；Amplify Hosting是最後選配，不要讓前端託管阻塞核心Demo。

## 7. API Gateway與CORS

使用HTTP API、Lambda proxy integration payload format 2.0，路由：

~~~text
GET  /api/health
GET  /api/options
POST /api/predict
POST /api/explain
POST /api/reveal
~~~

API Gateway統一設定：

~~~text
AllowOrigins: 實際前端的精確https網域；本機Demo階段另加http://localhost:3000
AllowMethods: GET, POST, OPTIONS
AllowHeaders: content-type, authorization
AllowCredentials: false
~~~

不要靠公開S3解決CORS。API Gateway啟用CORS後會處理preflight，避免與後端維護兩套衝突設定。整個LightGBM、SHAP及Bedrock回應必須短於HTTP API integration timeout；Bedrock timeout要更短，才能回傳fallback。

## 8. IAM最小權限

| Role | 只給這些能力 |
|---|---|
| Prediction Lambda | CloudWatch Logs、runtime bucket指定objects、指定Bedrock model/profile |
| Reveal Lambda | CloudWatch Logs、truth bucket指定object |
| Deploy/operator | ECR、Lambda、API Gateway部署；iam:PassRole只限上述roles |

Prediction role的policy不可出現truth bucket ARN。S3固定key只需 s3:GetObject；只有真的需要列舉時才加入受prefix限制的s3:ListBucket。Bedrock通常只需指定資源的 bedrock:InvokeModel。若使用SSE-KMS，還需精確key的kms:Decrypt及相容key policy。

前端不可持有AWS credentials，也不可直接呼叫Bedrock。

## 9. 必做驗收

依序完成並留下截圖／log：

1. 本機 smoke test全過。
2. Cloud /api/health 回200。
3. 預設三重案例：23個當下缺車、21個可評估、12個過平衡門檻、Top-10。
4. Cloud與本機同一批機率誤差不超過2e-7。
5. reveal預設案例為7/10；重複station ID不會灌高分。
6. Prediction role讀truth物件必須AccessDenied。
7. SHAP貢獻加總等於raw score，且不含答案欄。
8. Bedrock正常時回合法schema；關閉或故意timeout時仍顯示固定模板。
9. 非白名單Origin被擋，實際前端Origin可用。
10. 瀏覽器不出現AWS金鑰、S3 truth URL或答案欄。
11. 整次API呼叫低於API Gateway timeout；展示前先預熱一次。

若任何一項失敗，先保留本機前端＋本機API作為可用fallback，不在最後一刻重訓或改門檻。

## 10. 五分鐘Demo建議

1. 官方地圖只能顯示「現在」；我們要分流短暫與持續紅燈。
2. S3存放版本化模型輸入，truth權限分離。
3. 執行LightGBM預測，顯示門檻、有限清單與路線示意。
4. 點一站看SHAP原因；Bedrock只把原因整理成營運摘要。
5. 揭曉30分鐘後結果，說明逐站計分及模型限制。
6. 在AWS Console快速展示Lambda、API Gateway、S3權限及Bedrock呼叫紀錄。

## Definition of Done

只有同時符合下列條件才可在簡報寫「AWS／Bedrock已完成」：

- AWS URL可由前端呼叫，完整predict與reveal不依賴本機API。
- 模型輸入確實由私有S3取得，Prediction role不能讀truth。
- SHAP原因來自同一筆LightGBM推論。
- Bedrock在實際帳號及Region成功生成，且失敗時有fallback。
- Cloud與本機數字一致，沒有用6月重新調參。
- README的完成／未完成狀態已同步更新。

## AWS官方參考

- [S3安全最佳實務](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [S3資料加密](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingEncryption.html)
- [Lambda容器映像](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [Lambda限制](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)
- [API Gateway HTTP API Lambda整合](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html)
- [API Gateway HTTP API CORS](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-cors.html)
- [Bedrock InvokeModel](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html)
- [Bedrock模型存取](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
- [Bedrock Region支援](https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html)
- [Lambda execution role](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html)
- [IAM安全最佳實務](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS CLI SSO](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
