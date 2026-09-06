# 模型卡：YouBike 30 分鐘缺車持續風險

## 任務

對每一筆目前「精確缺車」（可借車數為 0）的有效快照，預測同一站點在 30 分鐘後是否仍精確缺車。模型輸出經 Platt 校正後的 0～1 風險機率。

## 訓練與封存

- 模型：LightGBM full，242 棵樹、68 個凍結特徵。
- 訓練資料：2026 年 1～3 月。
- 4 月前半校正機率，4 月 15～20 日選門檻，4 月 21～30 日只做 gate。
- 5 月未參與訓練、機率校正或門檻選擇。
- 6 月是一次性封存測試集，未用於調參。

## 凍結政策

- 平衡門檻（secondary / precision-first）：`0.5668120503425599`。
- 嚴格門檻（primary / p70-recall）：`0.6517772727012636`。
- Platt：`sigmoid(1.1206777095794678 × raw_score + 0.11242114752531052)`。

門檻名稱源自當時的預先宣告實驗政策；部署時不得依六月結果重新選門檻。

## 限制

這是歷史回放與決策輔助模型，不代表即時道路狀況，也不保證巡補後的因果效果。新站點會以 LightGBM 的 missing-category 路徑推論，應在正式上線後持續監測。站點維修、容量變更及資料延遲都可能影響結果。

Top-K 只是營運畫面上限，不是模型準確度。路線為風險優先後的相鄰距離串接示意，沒有納入車隊數、道路、車程成本或真實派車限制。

## 防資料洩漏

推論程式只用 freeze JSON 中的 68 欄白名單建立矩陣；`y_same_30`、`future_30_*` 與核對結果不會進入模型。測試集保留答案只為離線 parity test 與「揭曉結果」展示。

AWS 版使用兩個權限分離服務：Prediction Lambda role 只能讀 truth-free runtime objects，不能讀 truth bucket；Reveal Lambda role 才能讀指定 reference object。不過 `/api/reveal` 本身是歷史回放用的公開 Demo endpoint，所以應表述為「預測程序無權讀答案」，不是「答案對外完全不可取得」。

## SHAP 解釋

LightGBM 可對同一筆凍結輸入使用 `pred_contrib=True`，回傳 68 個特徵對 raw score 的貢獻與一個 base value。程式會硬性驗證加總結果等於 LightGBM raw score，再分別選出貢獻最大的 3 個推升因素與 2 個降低因素，最後依影響幅度排列。

SHAP 是對模型分數的解釋，不是真實世界因果關係；貢獻單位是 raw score，不能直接說成「增加幾個百分點機率」。中文特徵名、特徵值及方向由程式的固定白名單生成，不由生成式 AI 猜測。

## Bedrock 不是模型的第二次決策

Bedrock 只負責將已計算的 SHAP 與站點事實整理成可讀中文。送入只包含 allow-listed station facts 與最多 8 個已驗證 SHAP factors；Bedrock 只能從這些 feature IDs 選 1～3 個重點。最終文字由伺服器重新組裝，不讓 Bedrock 改風險、門檻、排名、路線或真值。

Bedrock 關閉、無權限、逾時、失敗或輸出不合規時，服務會回明確標示的固定範本，不影響 LightGBM 機率或 SHAP。只有 `provider: "amazon_bedrock"` 才代表實際 Bedrock 呼叫成功；`provider: "template"` 不得被報告成 Bedrock 生成。

## 部署狀態

AWS-ready 程式與部署骨架已完成，包含兩個 Lambda 容器、私有 S3、API Gateway、Amplify manual static、SHAP、Bedrock client/fallback、部署腳本與 CI。目前尚未有比賽 AWS 憑證，因此未實際部署、未真正呼叫 Bedrock，也未完成公開雲端驗收。這個狀態不影響凍結模型本身，但不可將 AWS-ready 說成 AWS 已成功上線。
