# 本機 Demo API 契約

前端預設連到 `http://127.0.0.1:8000`。這個介面刻意把「模型輸入」與「六月答案」分開，之後搬到 AWS 時可以直接對應 API Gateway 與兩個不同的 S3 路徑。前端正式網域以逗號分隔寫入 `UBIKE_ALLOWED_ORIGINS`，API 不使用任意來源萬用字元。

## `GET /api/health`

確認凍結模型與六月特徵已載入。

## `GET /api/options`

回傳五個經過核對的歷史回放情境、兩種凍結警示政策與預設值。可附加 `decision_time` 查詢該時點有哪些行政區候選。

## `POST /api/predict`

```json
{
  "decision_time": "2026-06-29T09:00:00+08:00",
  "district": "三重區",
  "policy": "balanced",
  "action_limit": 10
}
```

只讀取六月特徵與凍結模型，不讀答案。回傳：

- 當下所有0車候選；歷史長度不足者會明確標成未評分。
- 68項特徵經 LightGBM 與 Platt 校正後的持續缺車機率。
- 先通過凍結門檻、再套行動上限的清單，不會為了湊滿10站加入低風險站。
- 從最高風險站出發、再依鄰近距離串接的示意順序；這不是實際派車最佳化。

## `POST /api/reveal`

```json
{
  "decision_time": "2026-06-29T09:00:00+08:00",
  "station_ids": [673, 1526]
}
```

只有使用者按下揭曉後才讀取六月 reference。每個站只回傳「30分鐘後仍為0車」或「已恢復有車」，並以逐快照方式計算本案例 Precision；不使用 Episode 計分，也不虛構30分鐘後的精確車數。

## AWS 搬移邊界

| 本機 | AWS 對應 |
|---|---|
| `data/source/dynamic_red_empty_2026_06_input.parquet` | 私有 S3 `demo/inputs/` |
| `data/reference/` | 權限分離的 S3 `demo/truth/` |
| `backend/api.py` | Lambda 容器映像或 SageMaker endpoint 前的服務層 |
| `/api/*` | API Gateway 路由 |
| `frontend/` | Amplify Hosting 或 S3 + CloudFront |

AWS 只替換儲存、權限與服務位置；68欄特徵順序、Platt參數、模型檔及兩個門檻不可因搬遷而改變。模型檔、輸入檔與揭曉答案各有獨立環境變數，不互相推導路徑。
