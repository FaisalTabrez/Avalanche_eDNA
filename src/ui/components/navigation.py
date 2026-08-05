import streamlit as st

from src.ui.core.state_manager import StateManager


class Navigation:
    """
    Handles the application sidebar and navigation logic.
    """

    PAGES = {
        "Mission Control": "dashboard",
        "Analysis Hub": "analysis",
        "Data Explorer": "data_explorer",
        "Model Forge": "model_forge",
        "System Monitor": "system_monitor",
    }

    ICONS = {
        "Mission Control": "🚀",
        "Analysis Hub": "🧬",
        "Data Explorer": "🔍",
        "Model Forge": "⚡",
        "System Monitor": "🖥️",
    }

    @staticmethod
    def render():
        """Renders the sidebar navigation."""
        with st.sidebar:
            st.title("🏔️ Avalanche")
            st.caption("eDNA Biodiversity System")
            st.markdown("---")

            current_page = StateManager.get("current_page")

            # Navigation Menu
            selected_page = st.radio(
                "Navigate",
                options=list(Navigation.PAGES.keys()),
                format_func=lambda x: f"{Navigation.ICONS[x]} {x}",
                label_visibility="collapsed",
                index=(
                    list(Navigation.PAGES.keys()).index(current_page)
                    if current_page in Navigation.PAGES
                    else 0
                ),
            )

            if selected_page != current_page:
                StateManager.set("current_page", selected_page)
                st.rerun()

            st.markdown("---")

            # Quick Status in Sidebar
            status = StateManager.get("system_status")
            st.markdown("### System Status")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("CPU", f"{status.get('cpu', 0)}%")
            with col2:
                st.metric("RAM", f"{status.get('memory', 0)}%")

            db_status = "🟢 Online" if status.get("db_connected") else "🔴 Offline"
            st.caption(f"Database: {db_status}")

            st.markdown("---")
            st.caption("v2.0.0-MissionControl")
