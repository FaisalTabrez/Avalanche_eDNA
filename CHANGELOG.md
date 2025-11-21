# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2025-11-22
### Added
- Repository reorganization: created `scripts/windows/`, `archive/removed_misc/`, `tools/`, and `bin/` directories.
- Dry-run reports: `REORG_DRYRUN.md` and `DRYRUN_ONEOFFS_FULL.md` added.

### Changed
- Moved `start_report_system.bat` -> `scripts/windows/start_report_system.bat`.
- Moved `scripts/sra_integration_example.py` -> `examples/sra_integration_example.py`.
- Archived one-off files to `archive/removed_misc/` (e.g., `install_sra_toolkit.py`, `demo_analysis.py`).

### Notes
- Tests under `tests/` were left in place.
- No destructive deletions were performed; archived files are in `archive/removed_misc/`.
