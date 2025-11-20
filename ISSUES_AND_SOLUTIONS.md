# Issues and Possible Solutions

This document collects the issues discovered while auditing and running the test-suite, plus recommended fixes and future debugging notes. Use this as a single-source reference when triaging related problems.

## Summary
- Test run initially failed during import due to a broken `src/models/embeddings.py`.
- A Windows file-lock (PermissionError) occurred when tests removed a temporary directory containing `lineage_cache.db` created by `EnhancedTaxdumpResolver` in `src/clustering/enhanced_lineage.py`.
- Minor issues and warnings were observed (missing imports, tests that `return True` instead of asserting).

---

## High Priority Issues

- **Import-time NameError: `torch` not defined**
  - Location: `src/models/embeddings.py`
  - Symptom: Pytest collection fails with `NameError: name 'torch' is not defined`.
  - Likely cause: Partial/stubbed file referencing PyTorch symbols without importing them or without a proper fallback.
  - Recommended fix:
    - Provide a minimal, well-scoped implementation (or guarded imports) so import-time errors do not block test collection.
    - Example approaches:
      - Add safe `try/except ImportError` guards and fallback stub implementations that keep the public API stable for tests.
      - Keep the real PyTorch-backed implementation behind a factory that attempts to import torch at runtime.

---

## File-lock / PermissionError on Windows when deleting tempdir

- **PermissionError: [WinError 32] file is being used by another process**
  - Location: `tests/test_enhanced_taxonomy.py::TestPerformance::test_lineage_caching`
  - Symptom: TemporaryDirectory cleanup fails because `lineage_cache.db` remains open/locked.
  - Cause: SQLite connection(s) referencing that file are still open when the test tries to remove the tempdir. Windows prevents deletion of files with open handles.
  - Fix options (ordered by recommended approach):
    1. Robust connection lifecycle (recommended):
       - In `EnhancedTaxdumpResolver` (`src/clustering/enhanced_lineage.py`) create and reuse a single connection object for the cache (e.g., `self._cache_conn`) opened in `_init_cache_db()` and explicitly close it via a `close_cache()` method or via context manager / `__enter__/__exit__` pattern.
       - Ensure `_get_cached_lineage` and `_cache_lineage` use the same connection and do not open long-lived additional handles.
       - Call `resolver.close_cache()` from tests before TemporaryDirectory exit if tests create the resolver explicitly.
    2. Test-scoped accommodation (fast, less ideal):
       - If `taxdump_dir` is inside the system tempdir, use an in-memory SQLite DB (`":memory:"` or `file::memory:?cache=shared` URI) to avoid on-disk files in temp directories.
       - This approach was used temporarily during debugging but was reverted; prefer the robust lifecycle fix instead.
    3. Place cache files in a stable non-temp location (configurable):
       - Allow configuring a cache directory that defaults to `data/cache/` rather than the taxdump folder.

  - Additional notes:
    - Avoid relying on `__del__` to close DB handles on Windows — destructor timing is not reliable.
    - Prefer explicit close methods or use context managers where test code (or consumer code) can deterministically close resources.

---

## Other Issues & Warnings

- **Missing import: `json`**
  - Location: `src/preprocessing/sra_processor.py` (usage of `json.dump()` observed earlier)
  - Symptom: Runtime error if that code path is exercised.
  - Fix: Add `import json` at top of file (or ensure callers pass serializable objects).

- **Tests returning values instead of asserting**
  - Files: `final_test.py`, `test_comprehensive.py`, `test_fix.py`, `test_installation.py` produced `PytestReturnNotNoneWarning`.
  - Symptom: Warnings that tests are `return`ing a non-None value (future pytest error).
  - Fix: Replace `return True` with `assert` statements or remove `return` from test functions.

- **Potential heavy dependency issues**
  - Modules reference heavy optional dependencies: `torch`, `transformers`, `faiss`, `hdbscan`, `cutadapt`, `vsearch`.
  - Recommendation: Guard optional imports and provide light-weight stubs or mocks for test environments.
    - Use `try/except ImportError` and set an `IS_AVAILABLE` flag; provide `NotImplementedError` or stubs that explain missing features at runtime.

---

## Suggested Code Changes (Concrete)

1. EnhancedTaxdumpResolver resource management (high priority)

   - Add to `src/clustering/enhanced_lineage.py`:
     - `self._cache_conn: Optional[sqlite3.Connection] = None`
     - `_init_cache_db()` should set `self._cache_conn = sqlite3.connect(str(self.cache_db_path))` and run initialization SQL using that connection.
     - `_get_cached_lineage` and `_cache_lineage` should use `self._cache_conn.cursor()` instead of opening new connections.
     - Add `def close_cache(self):` which closes `self._cache_conn` and sets it to `None`.
     - Optionally implement `__enter__/__exit__` to allow `with EnhancedTaxdumpResolver(...) as resolver:` semantics.

2. Defensive imports and stubs for heavy libs

   - Example pattern for `src/models/embeddings.py`:

```python
try:
    import torch
except ImportError:
    torch = None

if torch is None:
    # minimal stub classes to satisfy tests
    class DNATransformerEmbedder: ...
else:
    # real PyTorch-backed implementations
    class DNATransformerEmbedder(torch.nn.Module): ...
```

3. Test improvements

   - Update tests that create a resolver in a tempdir to call `resolver.close_cache()` before the tempdir context exits (if you implement close_cache()).
   - Replace `return True` in tests with `assert True` or proper assertions.

---

## Commands & Checks

Run full test suite locally after fixes:

```bash
python -m pytest -q
```

Run a single failing test while iterating on fixes:

```bash
python -m pytest tests/test_enhanced_taxonomy.py::TestPerformance::test_lineage_caching -q
```

Lint & type-check suggestions:

```bash
# if using flake8/mypy
flake8 src tests
mypy src --ignore-missing-imports
```

---

## Prioritization
- P0: Fix `embeddings.py` import-time errors; ensure tests collect.
- P0: Fix cache-handling (explicit close / single connection) to eliminate Windows PermissionError.
- P1: Add defensive guards around optional heavy dependencies.
- P2: Clean test `return` statements and run linters.

---

## Notes for Future Debugging
- When Windows-specific PermissionError occurs on tempdir cleanup, search for open file handles to that file and ensure all DB connections are deterministically closed.
- Prefer deterministic cleanup paths: explicit close methods, context managers, or process-scoped temporary directories that are cleaned after closing resources.
- Keep heavy imports behind runtime guards so unit tests can run in minimal environments.

If you want, I can implement the robust cache lifecycle fix now (create a shared connection and add `close_cache()`), run the tests, and then create a PR. Reply with "Implement robust cache handling" to proceed.
