Repository Reorganization - Dry Run

Target layout (conservative, safe):
- `scripts/` : all Python utility scripts and helpers (most already in `scripts/`)
- `scripts/windows/` : Windows batch helpers (`*.bat`)
- `docs/` : developer documentation and design docs (move `ML_TRAINING_IMPLEMENTATION.md`)
- `tools/` : CLI helpers or optional tools
- `bin/` : small executable wrappers (if any)
- `archive/removed_misc/` : place to archive removed one-off files

Planned moves (dry-run - only conservative items moved in this operation):
- `start_report_system.bat` -> `scripts/windows/start_report_system.bat`
- `setup_report_management.py` -> `scripts/setup_report_management.py`
- `ML_TRAINING_IMPLEMENTATION.md` -> `docs/ML_TRAINING_IMPLEMENTATION.md`

Notes:
- Many tests, debug and demo scripts were previously deleted per prior requests; they are not included here.
- If any file in the list no longer exists, it will be skipped.
- After these moves we will run a quick smoke check and update `README.md` paths.
