import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ui.components.navigation import Navigation
from src.ui.core.state_manager import StateManager

# Import Modules (Lazy loading could be implemented here, but direct import is fine for now)
from src.ui.modules import (
    analysis,
    dashboard,
    data_explorer,
    model_forge,
    system_monitor,
)

# Page Configuration
st.set_page_config(
    page_title="Avalanche Mission Control",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Load custom CSS."""
    css_path = Path(__file__).parent / "styles" / "theme.css"
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # 1. Initialize State
    StateManager.initialize()

    # 2. Load Styles
    load_css()

    # 3. Render Navigation
    Navigation.render()

    # 4. Route to Module
    current_page = StateManager.get("current_page")

    # Main Content Container
    main_container = st.container()

    with main_container:
        if current_page == "Mission Control":
            dashboard.render()
        elif current_page == "Analysis Hub":
            analysis.render()
        elif current_page == "Data Explorer":
            data_explorer.render()
        elif current_page == "Model Forge":
            model_forge.render()
        elif current_page == "System Monitor":
            system_monitor.render()
        else:
            st.error(f"Page {current_page} not found.")


if __name__ == "__main__":
    main()
