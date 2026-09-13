# 私有資產與資料界線

以下檔案刻意不包含於公開提交。只能從經授權的私有來源取得；不可為了讓測試通過而將主辦資料推回 GitHub。

- `config/final_policy_freeze_before_may.json`
- `config/final_policy_freeze_full_dock_before_may.json`
- `config/protocol_frozen_before_june.json`
- `config/protocol_full_dock_frozen_before_june.json`
- `data/reference/june_all_eligible_decisions.parquet`
- `data/reference/june_full_dock_all_eligible_decisions.parquet`
- `data/source/dynamic_red_empty_2026_06.parquet`
- `data/source/dynamic_red_empty_2026_06_input.parquet`
- `data/source/dynamic_red_full_2026_06.parquet`
- `data/source/dynamic_red_full_2026_06_input.parquet`
- `data/stations/dim_station.csv`
- `model/lgbm_full.txt`
- `model/lgbm_full_dock.txt`
- `backend/supply_snapshot.json`
- `backend/supply_snapshot.sha256`
- `frontend/public/fleet-road-matrix.json`

`MANIFEST.json` 保留資產名稱與 SHA256；它不是資料或模型本身。公開版本已更新工具檔的雜湊；凍結模型與資料的原有雜湊保持不變。完整舊版 smoke test 需要先放回所有經授權的私有資產。

重現後端時，將經授權的資產放回上述路徑；模型、輸入、類別 schema 與校正參數需同一版本。`.env.example` 記錄可覆寫路徑，啟動腳本使用 `.venv`。供車快照固定為 6/29 09:00、1,562 站，需對應其 SHA256；道路矩陣的時間與站點座標也須一致，否則派車流程會拒絕規劃。

```powershell
.\scripts\start-local-app.ps1
# 下列全套測試需要私有資產；公開 CI 只跑 scripts/test-public-source.py。
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

AWS 部署腳本需經授權的 AWS profile、私有 S3 資產與完整快照；此公開版不能在缺少資產時直接重建 Live Demo。勿將 `key.cmd`、`.env`、憑證、部署輸出或 AWS 帳號暫時憑證提交。所列 AWS 服務是競賽允許的服務，不表示免費。

主辦規範要求公開程式碼不含 Access Key／Token／密碼，並保留使用過的 `.kiro`。兩份規範屬環境與服務限制，沒有提供資料集公開散布授權，因此本版採不附資料的界線。既有公開歷史並未在本次一般提交中清除。
