# YouBike 下一輪 Evaluation 任務書 v2

日期：2026-09-11。Repository：<https://github.com/spicyhoney/ubikepredict>。起始核對版本：`bdbea7a4c67d2bbaf028eb4950fab43141a61c00`。

這份文件供人類確認工作方向後，整份交給 agent 執行。它接續上一輪結果，**不是新實驗已完成的報告**。本輪結果另見 `EVALUATION_RESULTS_REVIEW_v2.md`；本文件已包含派工所需的問題、理由、規格、缺件處理與交付物。

## 1. 已知什麼，接著想回答什麼？

目前任務是：站現在已經沒車，預測同站 30 分鐘後是否仍沒車，再排出優先查看名單。

我們已重現 EVAL-2～4：只挑 3／5 站時「缺車最久優先」比完整模型準；挑 10／20 站時完整模型較準。相同時刻、相同已缺車步數內，模型仍有辨識力，尤其是剛觀察到缺車的站。Platt 校正則小幅改善機率誤差。

因此接下來要回答：

1. **這些額外辨識力，是哪類資料帶來的？** 需要完成尚未做的特徵消融。
2. **拿掉直接站點辨識資訊的版本，是否確實能重現？** no_station 的保存分數不錯，但目前還缺模型本身。
3. **長時間缺車規則與模型能否互補？** 能否保留長事件的優勢，又利用模型挑出容易被忽略的早期事件？

| 工作 | 預設優先級 | 需要訓練嗎？ | 要交出的答案 |
|---|---|---|---|
| R2-0：補齊評估來源與交付 | 先做，工程補件 | 不需要 | 別人能確認用的是哪個模型／任務書，並重生逐列結果 |
| R2-1：完成 A0～A5 特徵消融 | 主要研究工作 | 需要原始材料 | 每組資料有多少額外作用？ |
| R2-2：恢復 no_station 模型 | 與 R2-1 共用材料盤點 | 只有重訓新 A5 才需要 | 保存分數能否由原模型產生？它與新 A5 的差異是什麼？ |
| R2-3：長事件優先＋模型排序 | 可先做的低成本新實驗 | 不需要 | 混合規則在不同 K 是否比較好，代價是漏掉哪些站？ |
| R2-4：未見時間期間驗證 | 後續階段，資料齊才做 | 預設不重訓 | 結論在未用來設計規則的期間是否還成立？ |

EVAL-2～4 已完成，不必為了「多做實驗」無條件再跑一遍。R2-0 需要的回歸驗證、R2-1 新模型評估與 R2-3 新政策比較，才是本輪新增工作。

## 2. R2-0：先把「用什麼跑出來」交代完整

**為什麼要做？** 本輪程式已能重現，這項是讓別人換電腦、換環境後，也能確認自己確實用了同一批材料，避免 manifest 寫 A 模型卻因環境設定讀了 B 模型。

**執行規格：**

1. `scripts/run_evaluation.py` 建立 `FrozenYouBikeModel` 時，明確傳入當次 manifest 記錄的 `model_path`、`freeze_path`、`protocol_path`、`stations_path`；或改成記錄 engine 實際解析後的路徑。選一種一致做法並驗證，不能只記預設路徑。
2. 任務書改由明確的 `--spec-path` 或 repo 內文件位置提供；保存實際文件 hash，缺檔時明確報告，不能靜默略過。manifest 同時記錄 `backend/inference.py` hash、git HEAD 與 dirty 狀態，保留原本 runner/test/input hashes。
3. 補充「哪些產物隨 repo 提交、哪些由命令重生」。目前上游缺少已列出的三份 parquet 與 `eval4_scores.csv`，但可以重生；小型 CSV 可納入交付，大型 parquet 可保留於授權儲存位置，提供大小／hash／重生命令，不強迫把約 150 MB 的衍生檔提交 Git。
4. 報告對 June calibration intercept／slope 明確標成診斷用擬合，未套回正式機率。若列用六月正例率做的 constant prevalence baseline，標成事後描述參考，不稱為事前可部署模型；它的 ECE=0 不能單獨用來判斷模型好壞。
5. 做一個針對環境變數 override 的回歸檢查，證明實際載入路徑與 manifest 一致。沿用原基準參數重算必要結果，確認修正來源紀錄沒有改變評估母體、排序或數值。

**交付：** 小幅程式修正、回歸證據、完整 manifest、產物清單。這項不改模型、不改資料母體、不調六月門檻。

## 3. R2-1：完成尚未做的特徵消融

### 為什麼仍然需要它？

現在知道模型可以分辨同樣剛缺車的站，但它可能是在看車輛變化，也可能在記某個站的習慣。只看結果或 feature importance 不能分清楚。要讓模型少看一類資料、重新學一次，才知道那一類資訊在目前流程裡的增量作用。

每個版本都從相同 68 欄完整版本獨立移除，**不累加刪除**。A0 與所有消融版用同流程訓練；原 frozen 模型 F0 作歷史參照，不能代替新的 A0 控制組。

| 版本 | 拿掉什麼 | 剩幾欄 | 白話理由與結果解讀 |
|---|---|---:|---|
| A0 | 不拿掉 | 68 | 在同一個訓練流程下，建立完整資料的共同比較基準 |
| A1 | 3 個明確的已缺車時間編碼 | 65 | 看模型是否依賴「已經缺多久」這個摘要。三欄一起刪，避免同一資訊換個編碼留下。若下降，摘要有增益；若不降，仍可能從近期紅燈狀態推回時長 |
| A2 | 38 個近 30／60／90 分鐘的詳細狀態與變化 | 30 | 兩站都剛缺車，一個車數一路下降，另一個在 0／1／2 車間來回；模型是否需要這種差別？一起刪 lag、增減量、波動與紅燈代理，保留直接持續時間。下降才支持詳細動態在摘要之外還有幫助 |
| A3 | 8 個昨天／上週狀態與缺值旗標 | 60 | 已知道現在幾點、星期幾，還需要知道同站昨天／上週實際有沒有車嗎？保留日曆、移除歷史與其缺值旗標，避免把「時間」與「歷史觀測」混成同一問題 |
| A4 | 只拿掉 `station_id` | 67 | 分辨站點編號本身的作用。仍保留經緯度，所以沒下降不代表模型已經認不得站點 |
| A5 | 拿掉 `station_id`＋經緯度 | 65 | 重做既有 no_station 設定。與 A4 比較，可看刪 ID 後座標是否仍補上資訊；行政區、容量與同站歷史仍在，不能直接稱為新站泛化 |

### 這輪新結果讓哪些比較更重要？

**第一，看不同名單容量。** 原本 no_station 在小 K 比完整模型強，因此新 A4／A5 不能只報 K=10 最好的一格；K=3、5、10、20 全部保留。既有 no_station 分數與新 A5 分開命名，不混成同一次訓練結果。

**第二，看剛缺車的站。** EVAL-3 顯示最強證據來自 B0，故用上一輪既定的 B0～B3 表補看消融差異落在哪裡。例如拿掉近期變化後，B0 表現明顯下降，才比較支持近期變化對早期辨識有幫助。這仍是本輪 June 回溯分析，不變成事前確證性假說。

### 固定的訓練與評估規格

1. 先找回 1～3 月 features／labels、April 分割資料、原 builder／trainer／超參數／類別處理／stopping 規則。原 freeze 記錄已知時長訓練列 225,086、正例 99,802，可作 provenance 核對點，不為了湊數擅自刪資料。
2. 各版固定相同訓練列、時間切分、超參數、seed 與 early-stopping 規則；不單獨給某版更多搜尋預算。沿用原 seed；若未保存，先在 manifest 固定共同 seed=`20260910` 並註明。原驗證切分不明就列缺件，不拿六月來 early stop。
3. 先恢復並驗證 A0，再跑 A1～A5；新 A0 不一定與舊 F0 位元一致，須記錄原因，不能強行改流程湊舊分數。各版按實際保留特徵順序建立類別 schema，不使用測試標籤或新增 target encoding。
4. 主比較仍為 `P@10(Ai) − P@10(A0)`，負值代表拿掉該組後下降。另報各 K 的 selected／TP／FP／P@K／R@K、全體 eligible AP，以及 `P@10(A5) − P@10(A4)`。所有新模型用 raw margin 排序，確保同分規則一致。
5. 各主要差異以固定原始名單的每日 TP／selected 計數做成對 bootstrap：六月 30 日有放回抽 30 日，2,000 次，PCG64 seed=`20260910`，同一次兩方法用相同日期並保留重複日權重；取 2.5／97.5 百分位。只解釋日期抽樣變動，不假裝包含訓練 seed 變動。
6. B0～B3 定義仍為 steps=`0`、`1–2`、`3–5`、`≥6`。分開交付原全市 Top-10 的組別拆解，以及 `(datetime, duration_bin)` 組內各取最多 10 站的診斷；後者可達每時刻 40 站，不能當同一營運容量。保留各組候選、正例、入選數，以及候選 >10 的組數。
7. 第一輪不必重選各消融版門檻。若日後另比 threshold policy，才各自在 4/1～14 校正、4/15～20 依同規則選點、4/21～30 gate 不再調，沿用原 episode 邊界處理；不能把 F0 的 Platt／門檻硬套新模型。

**缺件時怎麼做：** 清楚列出缺哪份訓練表／設定、阻塞哪些版本。繼續 R2-0、可行的 R2-2 與 R2-3。把舊模型欄位填 0 或 missing 只能叫輸入擾動，不可代替本項重訓消融。

## 4. R2-2：把 no_station 從「保存分數」補成「可以載入的模型」

**白話理由：** 這版在小名單表現很好，很值得追；但只有一欄分數，還不能讓另一台電腦把相同輸入丟進模型、得到同樣結果。

**要拿回：** 原 no_station checkpoint、65 欄有序清單、類別 schema、前處理／校正設定、訓練版本及資料／split 記錄。history baseline 的原表與建表規則若可一起取得，也補上；不要把它當 no_station 必須等齊的條件。

**怎麼驗證：** 以同一份 truth-free June input 產生新分數，再接標籤。逐列對 `(datetime, station_id)` 比較 reference `p_lgbm_no_station`；報最大差異、不同 Top-K 名單與指標。若不一致，先查特徵、類別、校正、檔案版本，不以六月重新訓練來掩蓋差異。

**怎麼解讀：** 重現成功只代表這個舊模型與保存成績相符；要把差異歸因到特徵，仍應看 R2-1 的同流程 A0／A4／A5。既有 frozen no_station、保存分數、新重訓 A5 三者分開列。

**交付：** 模型來源及 hash、逐列差異、同 K 比較、是否重現的判定。找不到原模型就保留「保存分數評估」，不要補寫成成功。

## 5. R2-3：長時間缺車先看，其餘名額交給模型

### 為什麼提出這個實驗？

這輪看到兩個現象：長時間缺車的站本來就很容易持續，純 duration 在小 K 很強；模型則能分辨剛缺車的站，在較大 K 更好。

因此可以試一個簡單規則：**先看已觀察缺車至少 180 分鐘的站，剩下名額再按模型風險挑選。** 如果兩種資訊互補，可能兼顧兩端；但也可能把太多名額留給長事件，反而漏掉更值得看的早期站，所以必須比較。

這是看到六月結果後提出的新政策假說，即使 180 分鐘沿用既定 B3 邊界，也不代表已在未見資料上被驗證。本項不做 duration cutoff 或混合權重搜尋。

### 精確政策

方法名 `LONG_FIRST_F0`。每個 `datetime` 使用以下固定排序鍵，然後取 `min(K, n)`：

```text
1. long_flag = (red_duration_steps >= 6)，True 在前
2. p_lgbm_full 由高到低
3. station_id 由小到大
```

也就是：長事件超過 K 站時，長事件內依 F0 選滿 K；長事件少於 K 時全部入選，其餘以非長事件的 F0 分數補滿；沒有長事件時就等於 F0。不先套 balanced／strict 門檻，不動模型或 Platt。

### 先做一張名單差異表，理解原本怎麼輸贏

對原 F0 與 DURATION，在每個 K 分別列「兩者都選」、「只被 F0 選」、「只被 DURATION 選」的筆數、TP／FP、B0～B3 分布與長事件比例。兩者名單同長，獨有部分的名額也必須相同；TP 差可由兩個獨有集合直接重算。

這能檢查：小 K 的差距是否確實來自 F0 把容易持續的長事件換成較難判斷的早期站。若數據不支持，就不能把這個解釋寫成已知事實，也不因此臨時換 cutoff。

### 新政策怎麼比？

- 固定 K=`3,5,10,20`，全市按 `datetime` 分組；比較 F0、DURATION、LONG_FIRST_F0、RANDOM，以及有來源標示的 no_station 分數。完全相同 eligible key，selected 數必須相同。
- 固定同分規則沿用 ID 升冪；DURATION 另保留 100 seeds 的 label-blind 隨機同分敏感度。RANDOM 以排序後完整母體、PCG64 seeds 0～99 產生逐列優先值，在所有 K 共用；不挑最佳 seed。
- 報全部 K 的 selected、TP、FP、P@K、R@K；本項名單政策主表不拿 AP 當主要勝負。重點是 K=3／5 能否縮小 F0 的劣勢，以及 K=10／20 是否仍保有其增益。
- 預先報 `LONG_FIRST_F0 − F0` 與 `LONG_FIRST_F0 − DURATION`，全部 K 都呈現，沿用 R2-1 的成對按日 bootstrap。多個容量／對照是探索性分析，不宣稱已校正多重比較後的顯著發現。
- 補看新政策「換進／換出哪些 duration bin」，讓人知道提升是否伴隨早期站的漏失。方法若輸或效果小，也照實交付。

**交付：** 原方法交換名單分析、新政策的同 K 表／曲線、逐列 selections、來源與命令。用白話回答「規則與模型是否互補，有什麼取捨」。不因六月某格最好就替換 Demo。

## 6. R2-4：若要選新方法，再驗證未見期間

**白話理由：** 我們已經看過六月，再根據六月提出混合規則，很可能貼合這個月的特殊情況。要知道下個月是否仍有用，就需要先把方法固定，再看沒參與設計的新資料。

**先確認資料是否存在：** 由專案負責人指出六月之後、尚未被用於設計或選擇政策的連續至少 28 日區間，以及帶足前置 7 日歷史的輸入。記錄具體起訖日期與誰已看過結果；不可自己把七月或任何月份預設為乾淨 holdout。

**資料確認後才執行：** 在揭露該期間標籤前，保存模型／政策／特徵／K／母體條件與評估設定 hash。K=10 維持共同主要比較，K=3／5／20 分列輔助。沿用 exact-zero t+30 與相同 eligibility 原則，評估 F0、DURATION、固定 LONG_FIRST_F0，以及已完成且事先固定的 A0～A5／no_station；未拿到的模型列缺件，不補造。

資料 builder 必須確保特徵只用決策時點以前的資料，另報未知起點及 target 無效的排除量。這仍是有效 target 子母體的歷史評估，不自動代表所有線上站點。不要直接套只認六月的 `select_eligible_june`；建立等價、日期可設定的評估 selector，驗證在原六月會得到相同 key 後再評新期間，不修改 production selector。

未見期間共 D 個日曆日時，bootstrap 對該期間固定名單的每日計數有放回抽 D 日、保留重複日權重，沿用 2,000 次與預先記錄的 PCG64 seed；無列日補 0，不沿用寫死六月 30 日的抽樣陣列。期間一旦看過結果，不因成績不好便改參數後繼續稱同一盲測。新資料缺件只擋這個階段，其他工作正常交付。

## 7. 共用資料、指標與防止誤解的規格

所有路徑相對 repo root：

| 用途 | 路徑 |
|---|---|
| truth-free June input | `data/source/dynamic_red_empty_2026_06_input.parquet` |
| truth 與保存分數 | `data/reference/june_all_eligible_decisions.parquet` |
| F0 模型 | `model/lgbm_full.txt` |
| feature／Platt／政策 | `config/final_policy_freeze_before_may.json` |
| F0 類別 schema | `config/protocol_frozen_before_june.json` |
| 站點資料 | `data/stations/dim_station.csv` |
| 可用的評估起點 | `scripts/run_evaluation.py`、`tests/test_evaluation.py` |

- 明確使用 `load_demo_source(path=Path("data/source/dynamic_red_empty_2026_06_input.parquet"), mode="empty")`，再套 `select_eligible_june`；不要省略 path 而讀到 sealed cache。先推論，再以 `(datetime, station_id)` 一對一 join reference 的 `y_same_30`。
- June 固定 27,962 eligible、11,921 正例、420 時刻；source 92,882，尖峰 33,465，排除未知起點 5,503。當地時刻不轉移時區；target 必須是決策時間後 30 分鐘。現在 bikes=0 且 docks>0，母體有效性沿用 frozen 定義。
- 指標單位是「站點 × 決策時間」，同一站／episode 可以反覆入選，不說成不同受益站數或成功調度次數。兩個快照同為 0，不保證中間每一秒都缺車。
- P@K=`總TP/總selected`；R@K=`總TP/母體正例數`。用 pooled counts，不把各時刻 precision 無權重平均；只有自己的分母為 0 才 NA。AP 使用全部 eligible 分數與標籤，不先選 Top-K；分組 recall 用該組正例作分母。
- 對不需訓練的修正，先用現成 runner 作基準。需要新增的 runner 功能以最小增量完成，保持原評估可執行，不建立無關框架。

現成基準命令如下；它**只會跑既有 EVAL-2～4**，不會自動完成新 R2-1／R2-3：

```powershell
.venv\Scripts\python.exe -m unittest tests.test_evaluation -v
.venv\Scripts\python.exe scripts/run_evaluation.py --run-id <unique-run-id> --output-root outputs/evaluation --random-seeds 100 --bootstrap-repetitions 2000 --bootstrap-seed 20260910 --worktree-note '<actual status note>'
```

## 8. Agent 執行順序與完成條件

1. 記錄現況與 provenance，保留工作樹修改，不 reset／覆寫舊 run。先做 R2-0，同時盤點 R2-1／2 所需材料。
2. 資料未齊時先做 R2-3；原訓練材料到齊後才做 R2-1。恢復舊 no_station 與同流程重訓 A5 分開處理，不讓其中一項的缺件擋住所有分支。
3. R2-4 只在負責人確認未見期間與資料來源後啟動。不擅自使用額外付費算力，不把此文件當成部署／push／替換模型的要求。
4. 每項維持 `not_started / running / complete / missing_inputs / failed` 等真實狀態；只載入成功或只有保存分數，不能升格成完成新模型訓練。

建立 `outputs/evaluation/<new_run_id>/`，交付 `REPORT.md`、`metrics.csv`、逐列 predictions／selections、`run_manifest.json`、實際命令與 logs、圖表，必要時加 `models/` 與 `missing_inputs.md`。大型逐列檔可以放授權的穩定位置，manifest 必須給出路徑、大小、hash 與重生命令。每個 variant 保存實際有序特徵列表與 split／seed／超參數。

報告每項先用 2～4 句回答「原本問題、這次答案、證據、限制」，再放完整表格。未做的項目保留清楚缺件，不造數字、不把研究提案混入結果表。

> **可直接交給 agent：** 請依這份 v2 任務書工作，先補齊 R2-0 的 provenance 與可重生交付；盤點訓練與 no_station 材料，齊全後完成 R2-1／2。材料不足時先做固定的 R2-3 混合規則與名單交換分析。R2-4 等未見資料期間確認後再做。所有結果用白話解釋並交付可重算證據；不以六月調參、換門檻或挑最好的一格，不替換正式模型。缺件只暫停受影響工作，繼續其他分支。

## 附錄：A0～A5 精確移除欄位

使用 freeze `features.lgbm_full` 的 68 欄原順序，扣除下列指定群組。每版獨立扣除；執行時驗證不存在未知欄、重複欄或偷偷新增欄。這些欄名已靜態核對，下面只是未來重訓規格，沒有新訓練結果。

```yaml
full_feature_count: 68
variants:
  A0: {remove_group: null, remaining_count: 68}
  A1: {remove_group: explicit_duration, remaining_count: 65}
  A2: {remove_group: recent_30_60_90_dynamics, remaining_count: 30}
  A3: {remove_group: daily_weekly_history, remaining_count: 60}
  A4: {remove_group: station_id_only, remaining_count: 67}
  A5: {remove_group: existing_no_station, remaining_count: 65}
groups:
  explicit_duration:
    - red_duration_capped_6
    - red_duration_log1p_capped
    - duration_90plus
  recent_30_60_90_dynamics:
    - bikes_lag_30
    - bikes_lag_60
    - bikes_lag_90
    - docks_lag_30
    - docks_lag_60
    - docks_lag_90
    - bike_ratio_lag_30
    - bike_ratio_lag_60
    - bike_ratio_lag_90
    - dock_ratio_lag_30
    - dock_ratio_lag_60
    - dock_ratio_lag_90
    - delta_bikes_30
    - delta_bikes_60
    - delta_bikes_90
    - delta_docks_30
    - delta_docks_60
    - delta_docks_90
    - abs_delta_bikes_30
    - delta_bikes_ratio_30
    - delta_docks_ratio_30
    - bike_acceleration_30
    - dock_acceleration_30
    - bike_ratio_mean_90
    - bike_ratio_min_90
    - bike_ratio_max_90
    - dock_ratio_mean_90
    - dock_ratio_min_90
    - dock_ratio_max_90
    - lag_90_missing
    - suspected_large_jump_past
    - same_red_lag_30
    - same_red_lag_60
    - same_red_lag_90
    - opposite_red_lag_30
    - opposite_red_lag_60
    - opposite_red_lag_90
    - same_red_count_past_90
  daily_weekly_history:
    - bike_ratio_lag_1d
    - dock_ratio_lag_1d
    - bike_ratio_lag_7d
    - dock_ratio_lag_7d
    - lag_1d_missing
    - lag_7d_missing
    - same_red_lag_1d
    - same_red_lag_7d
  station_id_only:
    - station_id
  existing_no_station:
    - station_id
    - longitude
    - latitude
```

原始 `red_duration_steps`、`red_duration_log1p` 可留作 metadata／規則／分組，但不是 68 欄模型特徵；A1 刪掉三個編碼後不能把原始時長補回模型。`dynamic_episode_seq`、`episode_start_datetime`、`duration_unknown`、`is_first_observed_red`、`is_ongoing_red` 也不進模型。

A1 仍留有近期紅燈與車數代理，A2 仍留有持續時間與日／週歷史；A4 仍留座標，A5 仍留行政區、容量與同站歷史。解讀時保留這些限制。若單組結果顯示強烈替代，再另提 A1+A2 聯集移除作後續交互作用診斷，不自動加入本輪版本。
