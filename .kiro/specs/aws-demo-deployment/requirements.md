> 保留的部署階段規格；個人路徑已移除。舊授權與期限描述為當時紀錄，目前版本請見 README 與 docs/STATE.md。

# Requirements — YouBike AWS Demo 部署

日期：2026-09-12 更新。已 GO；本規格持續有效，實際進度以 STATE.md 為準。
唯一狀態來源：[docs/STATE.md](../../../docs/STATE.md)。工程規格供 Kiro／Codex 共用；是既有產品的部署收尾，非重做專案。
規格格式依 [Kiro Specs](https://kiro.dev/docs/specs/)。每项先用白話解釋，再給可驗收條件。

## R0｜使用正確的程式與授權

白話：GitHub 上的版本與目前電腦裡的修補不完全一樣；用錯版本，可能把已解決的部署錯誤帶回來。

- 必須在 <repository-root> 工作，保存 HEAD、git status、diff 與 stash 清單。
- 基線 HEAD=2c726b7e003c23ab7afa3c397ac8904382ffd683；八個 tracked 修改是既有成果，不能 reset／stash-pop／覆蓋；兩個 stash 保留。
- 收到使用者對本規格的 GO 前，只能閱讀、本地盤點與補交接資料。雲端部署、真實 Bedrock 呼叫、環境安裝與產品修補尚未開始。
- GO 可授權本規格內的必要修補、Docker／建置環境準備、已核對帳號內的具名 Demo 部署及有限次验收；不需逐步重問。
- 在创建 AWS 資源前必须有 profile、目標 account、允許 region、預算／可用額度、展示保留期限。新增服務架構、放寬 IAM、改研究、刪既有資料或超過預算要另對齊。
- 不 push／改 history；本機新文件不等於已提交到 GitHub。不同 agent 不同時修改本 repo。

## R1｜保留現有模型及研究結論

白話：上 AWS 是把同一套推論搬到雲端，不能為了讓 Demo 好看而偷換模型、門檻或答案。

- empty 與 full_dock 沿用各自凍結模型、68 欄順序、Platt 設定及門檻；模型／config／data 的大小與 hash 前後一致。
- 主要模型回答「當下0車，30分鐘後是否仍0車」。不改成新發缺車、數量預測、真實派車最佳化。
- App=行政區＋凍結門檻＋最多 K 站（預設10）；研究中的全市、不套門檻 Top-K 不是 App 指標。
- Prediction role 只能用當下／過去特徵，不能讀 truth；reveal 路由仍是公開的歷史結果展示。
- 沒有 R3、新訓練、新門檻、新策略、即時資料擷取的必要任务。混合排序仍是未採用的回溯候選。

## R2｜從另一台裝置完成雙模式展示

白話：評審看到的網站必須自己連雲端，不依靠開發者電腦在背景供應 API。

- 正式 Amplify HTTPS 網址載入成功；靜態 bundle 綁定實際 HTTPS ApiBaseUrl，沒有 localhost／127.0.0.1:8000／example.invalid。
- 缺車案例：2026-06-29 09:00+08，三重區、balanced、action_limit=10。
  當下23站、可評分21、過門檻12、入選10；reveal hits=7／10。
- 滿柱案例：2026-06-25 08:00+08，板橋區、balanced、action_limit=10。
  當下6站、可評分6、過門檻2、入選2；reveal hits=1／2。
- 點站可見正確 SHAP，切政策／情境／模式清除舊結果；重新 predict／reveal 可用。
- 派車情境只顯示条件推估；文字使用「30分鐘後已恢復」，不宣稱「自行恢復」或無介入；歷史分數不因模擬改寫。
- 在正式網址完成一次無痕或另一裝置操作；記錄實際環境，不把僅 HTTP 200 當成瀏覽器驗收。

## R3｜雲端推論與本地一致

白話：本機算對還不夠；容器的 Linux 套件、S3 載入與雲端設定也要產生同一個結果。

- Docker Linux daemon 可用；SAM build 實際完成兩個 Lambda 映像，Dockerfile 的 native import gate 通過。
- 全套本地 smoke 成功；目前基線6+58=64項，增加必要回歸測試後以全部通過為準。
- runtime 9檔、truth 2檔由指定 S3 keys 載入，大小與 SHA256 驗證成功；核心資產不可用時明確失敗，不用假資料替代。
- 双模式 cloud/local 同站機率最大絕對誤差 ≤2e-7；route／action 清單符合凍結政策。
- 冷／暖請求各留一次耗時與 CloudWatch 證據；操作須在現有服務 timeout 內得到正常結果。若第一次冷啟動失敗，要記錄並修正或明確設計預熱，不能只留下重試成功。

## R4｜Bedrock 是真正可降級的摘要服務

白話：LightGBM 決定風險，Bedrock 只幫忙整理原因。摘要壞掉時，預測仍要能展示。

- 啟用前修正 botocore Config：total_max_attempts=1，503／連線或讀取 timeout 不暗中重試。
- 新測試在 SDK HTTP 傳送層觀察送出次數，而不是只用 FakeClient 取代整個 SDK；不需真 AWS、不使用真 credentials。
- 本次演示保持 Bedrock read timeout 6 秒、connect 至多2秒、Lambda25秒；可用剩餘時間不足時回 template，不讓摘要耗盡 Lambda 回傳時間。參數组合改变必须重新驗證。
- 若 Bedrock timeout、AccessDenied、Throttling、無法解析或輸出不合規，API 回已驗證的 template 與 fallback_reason，不改概率／排序／SHAP。
- 成功驗收必须對正式端點實際得到 operational_summary.provider=amazon_bedrock；SDK FakeClient 或 template 不算。
- model/profile 在該帳號與 Region 可用，支援目前 Converse system／temperature／maxTokens／JSON要求。參數不支援時可做局部 adapter 修正，不能靠改 provider 欄假裝成功。
- IAM 使用精確 model/profile 與必要目的 foundation-model ARN；不以 Resource=* 解決 denied。

## R5｜保留可說得清楚的權限及費用邊界

白話：預測不能偷看答案，網站不能帶著 AWS 金鑰，示範也不能無限制消耗帳號額度。

- 兩個 S3 bucket 私有、Block Public Access、SSE-S3、Versioning；Prediction role 只有指定 runtime read，Reveal role 只有指定 truth read。
- 雲端記錄兩模式正常讀取及 Prediction role 讀兩個 truth object 實際 AccessDenied。
  若只有 IAM simulator，標記為政策模擬，不能冒充實際拒絕；受控同role診斷方法見 design D6。
- CORS 正確允許正式 FrontendUrl；非白名單 Origin 不取得允許該來源的 ACAO header。CORS 是瀏覽器規則，不能聲稱阻止 curl 或提供身分認證。
- 一般 API 維持3 RPS／burst10、explain維持1 RPS／burst1；先用 Bedrock-off 的小批請求驗證節流，限制測試次數。
- API Gateway 節流是 best-effort，並非硬性全帳號 Bedrock 1/s 上限；核對主辦規則。若要求硬限制，先回報需要另一個限流方案，不擅自加資料庫／佇列。
- Secret Access Key／Session Token 透過本機 AWS profile、SSO 或主辦憑證流程；不放聊天、repo、前端、spec、logs；也不收集或列印憑證值。
- 記錄此次 stack／SAM managed 資源與預計清理時間。AWS 預算警示不等於硬性停止；不啟用額外的常駐排程、壓力測試或未同意服务。

## R6｜可回查、可接手、可撤回

白話：明天的人要知道哪一步真的成功；雲端壞掉也知道如何退回本機或關閉摘要。

- 證據存在 work/aws-readiness 的部署子目錄，涵蓋非機密設定、精確命令／exit、artifact hash、資源輸出、驗收結果、已知限制。
- 更新 STATE，再更新 derived HANDOFF；實驗報告保留，PPT v12 仍為草稿，未要求就不改投影片。
- 第一次完整部署不使用 SkipAssetUpload／SkipFrontend／SkipVerify；不省略 final Bedrock驗收。
- deployment完成 ≠ cloud驗收完成。所有必要項成功才標「AWS＋Bedrock完成」；若只完成template架構，要清楚列尚缺Bedrock。
- 預設只預覽正式stack清理；永久刪除既有版本化bucket資料需明確授權。只清本次受控診斷的臨時資源可包含在GO的验收範圍。
