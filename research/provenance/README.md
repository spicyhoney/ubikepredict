# Legacy training provenance

These files are immutable source snapshots recovered from the original local
experiments.  They document how the dynamic exact-empty training tables and
the original frozen models were built.  Their SHA256 values match the hashes
recorded in `config/final_policy_freeze_before_may.json`:

- `run_dynamic_red_persistence_experiment_20260904.py`: original dynamic
  trainer (`code_sha256`).
- `run_current_red_persistence_experiment_20260902.py`: original cache builder
  (`cache_builder_sha256`).
- `train_youbike_model_v1_20260828.py`: upstream station and snapshot feature
  utilities used by the cache builder.

They are retained for audit, not used as the active next-round runner.  The
reproducible A0-A5 training entry point is
`scripts/run_evaluation_next.py`, which consumes the committed Jan-March
feature/label parquet files under `data/training/`.

The organizer's raw monthly CSV files are not duplicated in this repository.
They are not required to reproduce the A0-A5 fits from the committed derived
training tables.  Rebuilding those tables from raw snapshots requires the
authorized raw files and these provenance snapshots.
