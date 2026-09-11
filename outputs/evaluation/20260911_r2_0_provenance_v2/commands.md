# Commands

```powershell
C:\Users\a5140\OneDrive\桌面\hack2\.venv_youbike\Scripts\python.exe scripts\run_evaluation.py --run-id 20260911_r2_0_provenance_v2 --spec-path docs\YOUBIKE_EVALUATION_NEXT_v2.md --worktree-note "R2-0 through R2-3 implementation in progress; intended tracked changes present"
```

Verification:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_evaluation -v
.venv\Scripts\python.exe scripts\run_evaluation.py --run-id <new-run-id> --spec-path docs\YOUBIKE_EVALUATION_NEXT_v2.md
```
