# Commands

```powershell
C:\Users\stat-user\Desktop\hack2\ubikepredict\.venv\Scripts\python.exe scripts\run_evaluation.py --run-id 20260910_eval2_4_final --random-seeds 100 --bootstrap-repetitions 2000 --bootstrap-seed 20260910 --worktree-note "main tracks origin/main; intentional untracked files: scripts/run_evaluation.py, tests/test_evaluation.py"
```

Verification:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_evaluation -v
.venv\Scripts\python.exe scripts\run_evaluation.py --run-id <new-run-id>
```
