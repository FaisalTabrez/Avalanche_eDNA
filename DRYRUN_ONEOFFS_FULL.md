Deep dry-run report — candidate one-off/demo/debug/verify/install/example/test files
Date: 2025-11-22
Branch: chore/reorg-codebase

Summary:
- I ran a repository-wide read-only scan for filenames matching common demo/debug/verify/install/example/test patterns.
- This is a dry-run: no files were changed. Below are the candidates found and my suggested action for each.

Legend:
- [ARCHIVE] : move to `archive/removed_misc/` (safe, reversible)
- [KEEP] : leave in place (e.g., package tests or examples shipped intentionally)
- [REVIEW] : manual review suggested (could be moved into `scripts/`, `docs/` or `notebooks/`)

Candidates (path -> suggested action):
`./examples/ml_training_example.py` (example script)  -> [REVIEW]
	- Reason: useful example; consider keeping under `examples/` or moving into `docs/examples/` for discoverability.

`./final_test.py` (top-level one-off)  -> [ARCHIVE]
	- Reason: appears to be an ad-hoc test script. Archive to `archive/removed_misc/` for now (safe, reversible).

`./scripts/sra_integration_example.py` (example usage)  -> [REVIEW]
	- Reason: lives in `scripts/` but is an example; consider moving to `examples/` or adding a short README linking to it.

`./tests/test_enhanced_taxonomy.py`  -> [KEEP]
	- Reason: unit / integration tests — keep under `tests/` and leave to test workflow.

`./tests/test_ml_training.py`  -> [KEEP]

`./tests/test_system.py`  -> [KEEP]

Recommended next steps (dry-run -> action plan):
- Archive conservative one-offs: move `final_test.py` -> `archive/removed_misc/` (safe).
- Review example scripts: either keep under `examples/` or move `sra_integration_example.py` to `examples/` for consistency.
- Leave `tests/` content in place — do not archive/test removals unless you want to remove tests intentionally.

If you approve, I can (a) perform the safe archival of `final_test.py` to `archive/removed_misc/`, (b) move example files into an `examples/` folder (or leave them), and (c) update `README.md` to point to preserved examples and the `archive/removed_misc/` location.

