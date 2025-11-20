"""
Run Browser Page
"""
import streamlit as st
from pathlib import Path
import time
from src.utils.config import config as app_config

def render():
    """Browse and open runs stored under the configured runs directory"""
    st.title("Run Browser")

    # Base directory input
    default_root = str(Path(app_config.get('storage.runs_dir', 'runs')).resolve())
    root_dir = st.text_input(
        "Runs root directory",
        value=default_root,
        help="Root folder containing runs organized as <dataset_name>/<timestamp>"
    )

    if not root_dir:
        return
    base = Path(root_dir)
    if not base.exists():
        st.warning(f"Directory not found: {base}")
        return

    # Discover dataset folders
    datasets = sorted([d.name for d in base.iterdir() if d.is_dir()])
    cols_top = st.columns([2,2,2,1])
    with cols_top[0]:
        ds_filter = st.selectbox("Dataset", ["All"] + datasets)
    with cols_top[1]:
        search = st.text_input("Search (dataset/run)", "")
    with cols_top[2]:
        try:
            show_n = st.number_input("Show top N", min_value=5, max_value=500, value=20, step=5)
        except Exception:
            show_n = 20
    with cols_top[3]:
        refresh = st.button("Refresh")

    # Collect runs
    rows = []
    for ds in datasets:
        if ds_filter != "All" and ds != ds_filter:
            continue
        ds_path = base / ds
        try:
            for run_dir in ds_path.iterdir():
                if not run_dir.is_dir():
                    continue
                try:
                    mtime = run_dir.stat().st_mtime
                    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                except Exception:
                    mtime, mtime_str = 0, ""
                pr = (run_dir / 'pipeline_results.json').exists()
                tx = (run_dir / 'taxonomy' / 'taxonomy_predictions.csv').exists()
                nv = (run_dir / 'novelty' / 'novelty_analysis.json').exists()
                rows.append({
                    'dataset': ds,
                    'run': run_dir.name,
                    'path': str(run_dir.resolve()),
                    'modified': mtime_str,
                    'mtime': mtime,
                    'has_pipeline': pr,
                    'has_taxonomy': tx,
                    'has_novelty': nv,
                })
        except Exception:
            continue

    # Apply search filter
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r['dataset'].lower() or s in r['run'].lower()]

    # Sort and cap
    rows.sort(key=lambda r: r.get('mtime', 0), reverse=True)
    rows_view = rows[: int(show_n)] if show_n and len(rows) > show_n else rows

    # Render list
    if not rows_view:
        st.info("No runs found for the current filters.")
        return

    for i, r in enumerate(rows_view):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
            c1.markdown(f"**{r['dataset']} / {r['run']}**\n\n``{r['path']}``")
            c2.text(r['modified'])
            c3.markdown("results" if r['has_pipeline'] else "—")
            c4.markdown("taxonomy" if r['has_taxonomy'] else "—")
            if c5.button("Open", key=f"open_run_{i}"):
                st.session_state.prefill_results_dir = r['path']
                st.session_state.current_page_key = "results"
                st.rerun()
