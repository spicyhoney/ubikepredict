> 保留的部署階段規格；個人路徑已移除。舊授權與期限描述為當時紀錄，目前版本請見 README 與 docs/STATE.md。

# Tasks — YouBike AWS 部署與驗收

日期：2026-09-12 更新。狀態：已 GO；T0–T8 完成，AWS＋真實 Bedrock＋正式瀏覽器驗收通過。
先讀requirements與design。以下「完成」只能靠實際命令及結果勾選，不能因文件已寫或舊log存在而勾選。

## 這輪已完成的準備（2026-09-11）

- [x] 盤點HEAD與八項既有dirty修改／两個stash；MANIFEST24檔和S3 allow-list11檔核對。
- [x] 本地6+58=64項測試通過；前端lint通過。
- [x] SAM validate --lint及deploy -DryRun通過，未建立AWS資源。
- [x] 以example.invalid/prod做靜態build與ZIP檢查；Windows已知shutdown assertion由既有script受限處理。此ZIP不能上線。
- [x] 本地瀏覽器：缺車7/10、滿柱1/2、點站SHAP、本機fallback、派車示意、切政策清除舊結果再預測通過。
- [x] 產生這三份spec、使用者入口、更新STATE／HANDOFF；產品程式與凍結資產尚未更動。

證據：[work/aws-readiness](<local-audit-directory>)。
本次暫啟動的3000／8000程序已關閉。未驗AWS登入；只知本機有default、hackathon profile名稱。
下方 checkbox 為現況；詳細證據見 ../../../../aws-deployment/20260912-resume/ 與 docs/STATE.md。先前準備段落只代表 2026-09-11 歷史。

## 依賴與執行規則

T0 →（T1與T2可分檔並行）→ T3 → T4 → T5 → T6 → T7 → T8。
同一stack／IAM／部署腳本的mutations必須序列執行；不可同時由Kiro與Codex部署。
沒有GO時在T0的閱讀與本地盤點停止。帳號資料暫缺不阻止GO後的本地T1～T3，但不得越過T4進行雲端建立／付費呼叫。
本次範圍為demo；目標不是生產級營運系統或新研究。

## T0｜確認接手版本、範圍與待填設定

- [x] T0 完成。需求：R0、R1。

**為什麼：** 避免新clone漏掉八份已驗證修補，或登入到錯誤帳號。
**操作：** 用design D2的精確repo路徑，讀STATE／HANDOFF，保存HEAD、status、diff、stash；建立自己的deploy時間戳證據目錄。
核對使用者GO是否已存在；不把「準備交接」或本文件當成GO。列出缺的非機密設定，不索取Secret／Token。
**成功證據：** HEAD和dirty基線可追蹤，兩個stash保留；deployment-context.json有已知設定與未填欄，明確記go_received。
**失敗停點：** repo不符、其他agent正在改、model/data hash變動不明时先釐清；不reset、不自動pull。
**交付：** baseline.json、preexisting.patch、deployment-context.json（全部無秘密）。
**注意：** 本輪已有盤點基線，但下一執行輪須記下自己的起點，不必重訓或重新分析24個實驗。

## T1｜修Bedrock多一次重試及摘要逾時

- [x] T1 完成。依賴T0＋GO；需求R4。

**為什麼：** 一次摘要不應偷偷變兩次雲端呼叫，也不能在等摘要時耗盡Lambda回應時間。
**操作：** 按design D3，把SDK retry改total_max_attempts=1；把Lambda剩餘時間傳到摘要決策或做同等deadline guard，預留回傳margin。保留本地API相容性与現有template fallback。
**允许修改：** backend/bedrock_summary.py、必要的backend/api.py／lambda_handler.py與對應tests；如有被修改的程式MANIFEST項，只更新該項。
**驗證：** SDK transport注入503／timeout，send次數=1、provider=template；剩餘時間不足時send次數=0；正常合規JSON仍可回amazon_bedrock；錯誤不改risk／rank。
本機測試使用dummy credentials，不連AWS；完整smoke及相關回歸全部通過。
**成功證據：** bug reproduction與修補後結果、精確diff、測試命令／exit／log，模型／config／data hash不變。
**失敗停點：** 若需換模型框架、全域佇列／資料庫或權限放寬，回報方案，不擅自擴張。
**不可省略：** 只改一行Config而沒有觀察SDK實際send，不能宣稱retry行為已驗證。

## T2｜修正派車示意的一句話

- [x] T2 完成。依賴T0＋GO；需求R2、R1。

**為什麼：** 我們只知道30分鐘後已恢復，不知道是否有人運補，不能說「自行恢復」。
**操作：** frontend/app/page.tsx將該句改成「歷史上30分鐘後已恢復」；不變更模擬公式、模型或歷史結果。
**驗證：** frontend目錄執行pnpm run lint；瀏覽器選包含已恢復站的派車前N站，文字更新且歷史命中數不變。
**成功證據：** 小範圍diff、lint及UI文字紀錄。無需新增文字鏡像測試。
**失敗停點：** 不順便重做頁面或擴大派車模型。

## T3｜補Docker，真正建置Linux容器

- [x] T3 完成。依賴T1、T2；需求R3。

**為什麼：** Windows推論能跑不代表Lambda的Linux原生套件能載入。這是目前最明確的環境缺口。
**操作：** 按D2檢查／準備Docker官方安裝與Linux daemon；把既有SAM venv放當次PATH。需要重開機／正常安裝互動時列明原因與續跑命令。
使用既有lockfile／requirements，不更新依賴版本；如果pnpm shim仍依賴舊位置，重新產生前端依賴啟動器。
**驗證命令：**
```powershell
docker version
docker info --format '{{.OSType}}'
sam validate --template-file infra/template.yaml --lint --region us-east-1
sam build --template-file infra/template.yaml --build-dir .aws-sam/build --cached --parallel
.\scripts\smoke-test.ps1
.\scripts\package-frontend.ps1 -ApiBaseUrl https://example.invalid/prod -SkipInstall
```
**成功證據：** Docker Server/Linux、實際容器build log／image ID、native import gate成功、當輪smoke、ZIP結構／localhost掃描。
**失敗停點：** Docker缺失不能勾選通過；保留error、resume command。禁止拿Windows測試或舊CI檔案代替。
**提醒：** 最後的example.invalid ZIP仍是測試產物，不是T5要上傳的網站；T5必須重建。

## T4｜驗證帳號、主辦條件和費用範圍

- [x] T4 完成。依賴T0；部署前需T3成功。需求R0、R5。

**為什麼：** 能登入AWS不等於能建S3／Lambda／IAM，也不等於有Bedrock模型權限。
**操作：** 補齊profile、expected account、Region、AWS額度／預算、保留期限、主辦限制。使用者於本機完成SSO／短期憑證流程；執行STS核對Account。
只讀確認已存在同名stack狀態／ownership、可用Region、Bedrock model/profile候選、必要IAM服務範圍。
**成功證據：** 非機密deployment-context、STS target account核對結果、允許的資源範圍、模型待選／已選狀態。
**失敗停點：** 無效憑證、帳號不符、主辦禁止必要資源／cross-region、預算未知時，停雲端部分並集中報缺項。
**不得：** 把金鑰貼進對話／logs／repo；為了通過而設AdministratorAccess；自動接受付費模型訂閱。
**節流條件：** 核對1RPS究竟是路由目標還是硬性全帳號上限。若是硬上限，現有API Gateway配置不能證明符合，先回報最小補強方案。

## T5｜部署核心AWS與template模式

- [x] T5 完成。依賴T1～T4；需求R2、R3、R5。

**為什麼：** 先驗證模型、資料、API與公開網站的位置都接對，再處理Bedrock權限。
**操作：** 按D5 Phase A執行一次完整deploy，保持同一stack、stage與release，不加Skip旗標；保存CloudFormation events及outputs。
確認上傳11個精確資產、S3私有設定、Amplify manual deployment成功，bundle使用真ApiBaseUrl。
自部署腳本第一次自動API驗收就保存UTC時間、呼叫顺序與CloudWatch INIT／REPORT；用實際Init Duration識別冷啟動，不能等T7才把暖機請求當冷啟動。
**成功證據：** stack狀態、兩個Lambda與角色／S3／API／Amplify資訊、image digest／assets hash、實際FrontendUrl與ApiBaseUrl、雙模式基本驗收、template明確顯示。
**失敗停點：** 記精確最早失敗步驟与AWS action／resource／request ID；可在已核對範圍修部署bug並重跑，不反覆新建stack，不省驗收。
**重要：** T5完成仍不代表整個AWS＋Bedrock任務完成。

## T6｜用實際模型接通Bedrock

- [x] T6 完成。依賴T5、T1；需求R4、R5。

**為什麼：** 畫面出現摘要不代表Bedrock成功，本地template也能顯示文字。
**操作：** 確認模型支援目前Converse參數／輸出、帳號存取與完整ARN清單，再按D5 Phase B EnableBedrock部署。
只有必要且小範圍的模型參數adapter可修改；不能改predict或feature語義。
**驗證：** 正式API的empty與full_dock explain真實返回amazon_bedrock；CloudWatch有對應成功證據；記model/profile ID、Region、request ID／呼叫結果。
**成功證據：** 兩模式合規摘要＋provider、有限次fallback測試及恢復後成功；預測結果仍一致。
**失敗停點：** denied／unsupported參數／輸出不合規要保留真實原因；template繼續可用但T6不勾。需要超規格權限／付費訂閱時回報。

## T7｜把雲端完整驗收補齊

- [x] T7 完成。依賴T6；需求R2～R5。

**為什麼：** 部署腳本沒有執行真正瀏覽器，也沒有證明Prediction role實際讀不到答案。
**操作：** 用D6矩陣逐項執行；verify-cloud必須傳LocalBaseUrl及RequireBedrock。正式cloud與本機不能指向同一endpoint以假造parity。
**必须記錄：**
1. empty 23/21/12/10，reveal7/10；full_dock6/6/2/2，reveal1/2。
2. 兩模式cloud/local最大概率差≤2e-7；SHAP sum check及feature allow-list通過。
3. 同Prediction role對兩truth keys的實際AccessDenied＋runtime成功對照；只用simulator要標未完成实际驗證。
4. 真origin正向CORS與非法origin不獲允許；stage/route節流值＋有限小批行為觀察，禁止宣稱絕對全域上限。
5. 正式網址的browser predict／點站SHAP／reveal／模式與政策切換／模擬；另一台裝置或無痕。
6. 檢閱從T5／T6第一次自動驗收保存的cold/warm與CloudWatch證據；若未捕捉cold，另安排受控驗證，不把T7普通请求冒充冷啟動。template降級與恢复Bedrock後真provider；前端沒有secret或placeholder API。
**成功證據：** validation.json逐項pass＋actual／policy_simulation等證據層级、logs／截圖。IAM診斷若使用臨時函式，記exact ARN並清除此run的probe。
**失敗停點：** 任一必要項未通就保留未完成，不能用「看起來正常」全部勾選；把具體阻礙帶回owner。

## T8｜收尾、展示排練與交接

- [x] T8 完成。依賴T7；需求R6。

**為什麼：** 部署完成後，使用者需要可以展示與日後清理的結果，而非一堆成功log。
**操作：** 產生deployment-result.md，內容包括非機密target、版本與hash、URL、案例結果、Bedrock真實證據、延遲、限制、費用追蹤、資源清單與清理預覽命令。
檢查git diff範圍與模型資產hash，保留stashes；更新docs/STATE.md，再更新docs/HANDOFF.md和本機AWS交接入口的「實際結果」。
用正式網址排練主情境，避免未預期多個點站造成explain429。投影片更新待使用者要求，不自動改v12或宣稱已定稿。
**成功證據：** 使用者可開啟正式網址及驗收文件，下一人能重跑／恢復／清理；臨時測試process/probe已關閉，未清其他資源。
**交付文字格式：** 已完成哪些 → 真實驗收 →網址／證據 →還有哪些限制 →清理方式。
**不得：** 未獲授權push、刪除既有bucket版本內容、設定自動刪除排程，或把只到T5當完整完成。
