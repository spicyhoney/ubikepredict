# Demo API 契約（本機與 AWS 共用）

本機前端預設連到 `http://127.0.0.1:8000`；靜態 Amplify 建置必須透過 `NEXT_PUBLIC_API_BASE_URL` 指向真實 HTTPS API Gateway URL，不允許回退至 localhost。

AWS 使用 API Gateway HTTP API payload format 2.0。POST 請求必須使用 `Content-Type: application/json`（可附 `charset=utf-8`），JSON body 不得超過 64KB。錯誤回應為 JSON；Lambda 版會帶穩定 `code`，本機開發伺服器目前只保證 `error` 文字，例如：

```json
{
  "error": "decision_time 不可為空",
  "code": "INVALID_REQUEST"
}
```

## `GET /api/health`

確認預測服務、凍結模型與 truth-free 六月輸入已載入。這個路由在 Prediction Lambda，不會讀取 truth bucket。

## 模式參數

所有主要請求共用 `mode=empty|full_dock`：

- `empty`（預設）：當下0車，預測30分鐘後是否仍0車；這是主模式。
- `full_dock`：當下0空位，預測30分鐘後是否仍0空位；這是輔助模式。

## `GET /api/options`

回傳指定模式已核對的歷史回放情境、兩種凍結警示政策與預設值。使用 `?mode=full_dock&decision_time=...` 可查滿柱輔助模式的時點與行政區；不傳 `mode` 就是 `empty`。

## `POST /api/predict`

```json
{
  "mode": "empty",
  "decision_time": "2026-06-29T09:00:00+08:00",
  "district": "三重區",
  "policy": "balanced",
  "action_limit": 10
}
```

`mode` 可為 `empty` 或 `full_dock`，省略時預設 `empty`。`action_limit` 可為 1～25，省略時預設 10。`policy` 可為 `balanced` 或 `strict`，門檻由模式各自的凍結設定決定。

此端點只讀取 truth-free 輸入、凍結模型、設定與站點資料，不讀六月答案。回傳：

- `empty` 的當下0車候選，或 `full_dock` 的當下0空位候選；歷史長度不足者明確標示未評分。
- 各模式的 68 特徵經對應 LightGBM 與 Platt 校正後的持續失衡機率。
- 先通過凍結門檻、再受 `action_limit` 限制的行動清單，不會硬湊名額。
- 從最高風險站開始、再依相鄰距離串接的巡補順序示意；這不是實際派車最佳化。

## `POST /api/explain`

```json
{
  "mode": "empty",
  "decision_time": "2026-06-29T09:00:00+08:00",
  "district": "三重區",
  "station_id": 673,
  "policy": "balanced",
  "action_limit": 10
}
```

`action_limit` 與 predict 一樣可為 1～25，省略時預設 10。保留它是因為路線順位與「是否在行動清單」會受行動上限影響。只能解釋該時點、該行政區內有完整歷史的站點。

回應結構摘要：

```json
{
  "mode": "empty",
  "decision_time": "2026-06-29T09:00:00+08:00",
  "target_time": "2026-06-29T09:30:00+08:00",
  "station": {
    "station_id": 673,
    "station_name": "站名",
    "district": "三重區",
    "risk_probability": 0.72,
    "red_duration_display": "已持續30分鐘"
  },
  "shap": {
    "base_value_raw": -0.2,
    "raw_score": 0.9,
    "sum_error": 0.0,
    "factors": [
      {
        "feature": "red_duration_capped_6",
        "label": "缺車持續時間",
        "value_display": "30分鐘",
        "contribution": 0.42,
        "direction": "increase"
      }
    ],
    "disclaimer": "貢獻值是LightGBM原始分數的影響量，不是機率百分點，也不代表因果關係。"
  },
  "operational_summary": {
    "provider": "template",
    "text": "由伺服器依已驗證事實產生的摘要。",
    "fallback_reason": "bedrock_disabled"
  }
}
```

SHAP 由伺服器對同一筆 LightGBM 輸入計算。Bedrock 僅從 allow-listed SHAP feature IDs 中選擇重點，最終中文摘要由伺服器用白名單事實組裝，不能改機率或覺得哪一站應被派車。

- 只有實際 Bedrock 呼叫成功且輸出通過驗證時，`provider` 才是 `amazon_bedrock`。
- 本機、Bedrock 關閉、逾時、失敗或輸出不合規時，`provider` 為 `template`，並可帶 `fallback_reason`。
- Bedrock 失敗不影響 LightGBM 風險機率與 SHAP 原始解釋。

## `POST /api/reveal`

```json
{
  "decision_time": "2026-06-29T09:00:00+08:00",
  "station_ids": [673, 1526]
}
```

只有使用者按下揭曉後才由 Reveal 服務讀取對應的六月 reference。`empty` 回傳 `still_empty`，`full_dock` 回傳 `still_full_dock`，並共同提供 `still_imbalanced`。本案例 Precision 以逐站逐快照方式計算；不使用 Episode 計分，也不虛構30分鐘後的精確車數或空位數。

在 AWS 上，此路由為公開 Demo API，但由獨立 Reveal Lambda 與 IAM role 處理。Prediction Lambda role 完全沒有 truth bucket 讀取權。因此報告可說「預測與揭曉權限隔離」，不可說「truth 對外完全無法取得」。

## AWS throttle 與 CORS

- 全體 API 預設限速：3 requests/second，burst 10。
- `POST /api/explain` 額外限速：1 request/second，burst 1，符合主辦 2026-07-22 規範的 Bedrock 每秒最多 1 次要求。
- CORS 允許真實 Amplify HTTPS origin、本機 `http://localhost:3000`，及可選的精確 HTTPS origin override。
- 瀏覽器只呼叫 API Gateway，不直接讀 S3 或 Bedrock。

## 本機與 AWS 對應

| 本機 | AWS 對應 |
|---|---|
| 兩種模式的 truth-free Parquet、model、config，及共用站點 CSV | private runtime S3 |
| 兩個六月 reference Parquet | 權限分離的 private truth S3 |
| `backend/api.py` | 核心 DemoService，本機 HTTP 與 Lambda 共用 |
| `backend/lambda_handler.py` | 兩個 API Gateway v2 Lambda handlers |
| `/api/*` | API Gateway HTTP API routes |
| `frontend/dist/client` | Amplify manual static deployment |

AWS 只替換儲存、權限與服務位置；兩種模式共用這些端點與同一部署腳本，只增加 S3 模型／config／input／truth 資產。各自的 68 欄順序、Platt 參數、模型檔與兩個門檻都不變，不需重訓。
