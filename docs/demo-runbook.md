# Demo 操作稿（約60～90秒）

## 正式版操作順序

1. 開場先說：「官方畫面已能顯示現在是否0車；我們處理的是紅燈出現後，辨識哪些站30分鐘後仍可能是0車。」
2. 確認模式為「缺車持續（主要）」，使用預設「6/29 09:00 三重區」、平衡模式，按「執行預測」。指出當下23站0車，21站可評分，12站超過門檻，有限行動清單最多列10站。
3. 說明地圖與右側順序是「先看風險，再以相鄰距離串接」的巡補清單示意，不宣稱是實際最短派車路徑。
4. 點選一個可評分站點，展示 SHAP 的「推升風險」與「降低風險」因素。接著看營運摘要：

   - 若畫面真的標示 `Amazon Bedrock`，再說：「Bedrock 只從這些白名單 SHAP 因素選重點，最後文字由伺服器用已驗證事實組裝，不會改機率。」
   - 若標示「模型原因摘要／本機備援」，就說：「目前 Bedrock 未啟用或已 fallback，SHAP 仍是真實模型貢獻，摘要使用固定範本。」不可假裝是 Bedrock 產生。

5. 按「揭曉09:30結果」。本案例逐站命中7/10；這是六月未參與訓練或調參的歷史回放，不是 Episode 分數。
6. 收尾說：「預測與揭曉在 AWS 使用兩個 Lambda roles；Prediction role 不能讀 truth bucket。Reveal 則是使用者按鈕後才呼叫的公開 Demo endpoint。」

## 滿柱輔助模式（時間足夠再展示）

1. 切換到「滿柱持續（輔助）」，使用預設「6/25 08:00 板橋區」平衡政策。
2. 說明它使用獨立凍結 LightGBM、Platt 與門檻，但共用同一個 API、AWS 架構、SHAP／Bedrock 解釋面板與部署腳本。
3. 固定案例有6站當下0空位、6站可評分、2站過平衡門檻；揭曉為1/2。這只是流程回放，不是整體 Precision。
4. 六月整體回溯為：平衡版 Precision 47.71%、Recall 17.20%；嚴格證據版 Precision 56.79%、Recall 10.14%。因此只定位成輔助證據，不取代缺車主模式。
5. 六月未參與訓練、校正或門檻選擇，但快取已在專案過程中存在；報告時說「凍結後回溯核對」，不說「全新盲測」。

## 比賽現場 AWS 說法

只有在公開 Amplify HTTPS 網址、API Gateway、兩個 Lambda、私有 S3、SHAP 與實際 Bedrock 呼叫都通過驗收後，才說「AWS 版已完成」。

如果當下還沒有競賽 AWS 憑證，應說：「部署程式、SAM 模板、容器、S3 驗證、SHAP、Bedrock client、fallback 與前端都已 AWS-ready；真實帳號資源與雲端驗收尚未執行。」

## 備用情境

- 高負載：6/22 08:00 板橋區，展示大量紅燈如何縮成有限清單。
- 寫實案例：6/10 19:30 板橋區，展示命中與誤報並存。
- 補充案例：6/23 19:30 板橋區，單一案例很好看，但必須主動說明不能代表六月整體 Precision。

## 如雲端當場失敗

1. 不重訓、不改門檻、不用六月調數字。
2. 先用實際參數重跑 `scripts/verify-cloud.ps1 -StackName "ubikepredict-demo" -Region "實際Region" -Profile "實際Profile" -Origin "實際Amplify網址"`。它會查 API 固定案例、SHAP／摘要、Amplify 頁面與 CORS；若部署時要求 Bedrock，再加 `-RequireBedrock`。S3／IAM 權限與 throttle 另依 AWS 交接清單查看 CloudWatch 與手動驗收。
3. Bedrock 單獨失敗時可繼續展示 SHAP 與 template fallback；這是預先設計的容錯。
4. 若雲端整體無法恢復，使用 `.\scripts\start-local-app.ps1` 啟動本機備援，並誠實告知評審當前是本機模式。

`verify-cloud.ps1` 會在同一次驗收同時檢查缺車主案例與滿柱輔助案例，不需另一支 AWS 腳本或現場重訓。
