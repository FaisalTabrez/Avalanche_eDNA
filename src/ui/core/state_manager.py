from typing import Any, Dict, Optional

import streamlit as st


class StateManager:
    """
    Centralized management for Streamlit session state.
    Ensures type safety and default values for app-wide state.
    """

    DEFAULTS = {
        "current_page": "Mission Control",
        "analysis_active": False,
        "current_analysis_id": None,
        "user_settings": {"theme": "dark", "notifications": True},
        "recent_activities": [],
        "system_status": {"cpu": 0, "memory": 0, "disk": 0, "db_connected": False},
    }

    @staticmethod
    def initialize():
        """Initialize session state with defaults if not present."""
        for key, value in StateManager.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        """Set a value in session state."""
        st.session_state[key] = value

    @staticmethod
    def update_nested(key: str, subkey: str, value: Any):
        """Update a value inside a dictionary in session state."""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            st.session_state[key][subkey] = value
        else:
            # Initialize if missing or not a dict (though initialize() should handle this)
            st.session_state[key] = {subkey: value}

    @staticmethod
    def clear():
        """Clear all session state."""
        st.session_state.clear()
        StateManager.initialize()
