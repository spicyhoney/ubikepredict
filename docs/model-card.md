# 模型卡：YouBike 30 分鐘持續失衡風險

## 任務與優先級

本專案有兩個獨立凍結模式，API 使用 `mode=empty|full_dock`：

- `empty`（主模式）：當下可借車數為 0，預測同站在30分鐘後是否仍為0車。
- `full_dock`（輔助模式）：當下可還空位為 0，預測同站在30分鐘後是否仍為0空位。

兩者都輸出經 Platt 校正的 0～1 風險機率，但是獨立模型、獨立校正公式與獨立門檻。`full_dock` 是補足「無位可還」的輔助視角，不取代缺車主故事。

## 訓練與封存

- 缺車模型：LightGBM full，242 棵樹、68 個凍結特徵。
- 滿柱模型：LightGBM full，114 棵樹、68 個凍結特徵。
- 訓練資料：2026 年 1～3 月。
- 4 月前半校正機率，4 月 15～20 日選門檻，4 月 21～30 日只做 gate。
- 5 月不參與重訓、機率校正或門檻選擇。
- 6 月不參與訓練、校正或選門檻，只做凍結結果核對與歷史回放。但專案中已存在六月快取，特別是 `full_dock` 應稱為「六月回溯核對」，不應宣稱為新取得、從未開封的前瞻盲測。

## 凍結政策

| 模式 | Platt 校正 | 平衡門檻 | 嚴格門檻 |
|---|---|---:|---:|
| `empty` 缺車主模式 | `sigmoid(1.1206777095794678 × raw_score + 0.11242114752531052)` | `0.5668120503425599` | `0.6517772727012636` |
| `full_dock` 滿柱輔助 | `sigmoid(0.8807565569877625 × raw_score - 0.15054720640182495)` | `0.42059604096412656` | `0.48107434749603273` |

平衡門檻對應 `precision_first`，嚴格門檻對應 `p70_recall`。門檻名稱沿用當時預先宣告的實驗政策；滿柱嚴格政策在門檻選擇集並未真正達到70% Precision，因此不能根據名稱宣稱它有70%。部署時不得依六月結果重選門檻。

## 六月回溯結果

| 模式／政策 | 有效決策數 | Precision | Recall | 解讀 |
|---|---:|---:|---:|---|
| `empty` 平衡 | 27,962 | 65.84% | 21.26% | 主 Demo 政策 |
| `empty` 嚴格 | 27,962 | 68.56% | 7.00% | 警示少、較保守 |
| `full_dock` 平衡 | 3,776 | 47.71% | 17.20% | 輔助監看模式 |
| `full_dock` 嚴格 | 3,776 | 56.79% | 10.14% | 嚴格證據版，仍不是70% |

上表都是逐站逐快照計分，不使用 Episode 命中，也不是 Top-K 案例分數。單一 Demo 案例的 7/10、1/2 或 2/2 只用來展示回放流程，不得取代六月整體 Precision。

## 限制

這是歷史回放與決策輔助模型，不代表即時道路狀況，也不保證巡補後的因果效果。新站點會以 LightGBM 的 missing-category 路徑推論，應在正式上線後持續監測。站點維修、容量變更及資料延遲都可能影響結果。

Top-K 只是營運畫面上限，不是模型準確度。路線為風險優先後的相鄰距離串接示意，沒有納入車隊數、道路、車程成本或真實派車限制。

`full_dock` 回溯 Precision 明顯低於缺車主模式，因此產品上只能當補充證據，不應與 `empty` 使用同等強度的準確度宣稱。

## 防資料洩漏

推論程式只用對應 freeze JSON 中的 68 欄白名單建立矩陣；`y_same_30`、`future_30_*` 與核對結果不會進入任一模型。測試集保留答案只為離線 parity test 與「揭曉結果」展示。

AWS 版使用兩個權限分離服務：Prediction Lambda role 只能讀兩種模式的 truth-free runtime objects，不能讀 truth bucket；Reveal Lambda role 才能讀指定 reference objects。不過 `/api/reveal` 本身是歷史回放用的公開 Demo endpoint，所以應表述為「預測程序無權讀答案」，不是「答案對外完全不可取得」。

## SHAP 解釋

LightGBM 可對同一筆凍結輸入使用 `pred_contrib=True`，回傳 68 個特徵對 raw score 的貢獻與一個 base value。程式會硬性驗證加總結果等於 LightGBM raw score，再分別選出貢獻最大的 3 個推升因素與 2 個降低因素。滿柱模式使用對應的「滿柱／空位」中文語意，不把同方向紅燈誤譯為缺車。

SHAP 是對模型分數的解釋，不是真實世界因果關係；貢獻單位是 raw score，不能直接說成「增加幾個百分點機率」。中文特徵名、特徵值及方向由程式的固定白名單生成，不由生成式 AI 猜測。

## Bedrock 不是模型的第二次決策

Bedrock 只負責將已計算的 SHAP 與站點事實整理成可讀中文。送入只包含 allow-listed station facts 與最多 8 個已驗證 SHAP factors；Bedrock 只能從這些 feature IDs 選 1～3 個重點。最終文字由伺服器重新組裝，不讓 Bedrock 改風險、門檻、排名、路線或真值。

Bedrock 關閉、無權限、逾時、失敗或輸出不合規時，服務會回明確標示的固定範本，不影響 LightGBM 機率或 SHAP。只有 `provider: "amazon_bedrock"` 才代表實際 Bedrock 呼叫成功；`provider: "template"` 不得被報告成 Bedrock 生成。

## 部署狀態

AWS-ready 程式與部署骨架已完成，兩種模式共用 API Gateway、Prediction／Reveal Lambda、S3 分層、Amplify、SHAP、Bedrock client/fallback 及同一支部署腳本；`full_dock` 只新增自己的 model、config、truth-free input 與 truth 資產，不需現場重訓。目前尚未有比賽 AWS 憑證，因此未實際部署、未真正呼叫 Bedrock，也未完成公開雲端驗收。這個狀態不影響凍結模型本身，但不可將 AWS-ready 說成 AWS 已成功上線。
