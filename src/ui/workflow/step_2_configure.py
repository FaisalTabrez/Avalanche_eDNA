"""
Step 2: Configuration Component
Unified configuration for analysis, model training, and dynamic scaling
"""

from typing import Any, Dict

import streamlit as st

# Configuration presets
PRESETS = {
    "quick_analysis": {
        "name": "Quick Analysis",
        "description": "Fast scan for basic statistics and quality metrics",
        "icon": "⚡",
        "analysis": {
            "quality_analysis": True,
            "diversity_analysis": False,
            "taxonomy_analysis": False,
            "novelty_detection": False,
            "max_sequences": 10000,
        },
        "model": {"mode": "none"},
        "scaling": {"enabled": False},
        "estimated_time": "2-5 minutes",
        "memory_usage": "1-2 GB",
    },
    "full_edna": {
        "name": "Full eDNA Pipeline",
        "description": "Complete analysis + model training + dynamic scaling",
        "icon": "🧬",
        "analysis": {
            "quality_analysis": True,
            "diversity_analysis": True,
            "taxonomy_analysis": True,
            "novelty_detection": True,
            "max_sequences": 0,  # All
        },
        "model": {
            "mode": "train_new",
            "architecture": "contrastive",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "scaling": {
            "enabled": True,
            "auto_configure": True,
            "min_clusters": 10,
            "max_clusters": 10000,
        },
        "estimated_time": "15-30 minutes",
        "memory_usage": "4-8 GB",
    },
    "training_only": {
        "name": "Model Training Only",
        "description": "Train contrastive learning model without full analysis",
        "icon": "🤖",
        "analysis": {
            "quality_analysis": True,
            "diversity_analysis": False,
            "taxonomy_analysis": False,
            "novelty_detection": False,
            "max_sequences": 0,
        },
        "model": {
            "mode": "train_new",
            "architecture": "contrastive",
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
        },
        "scaling": {"enabled": True, "auto_configure": True},
        "estimated_time": "10-20 minutes",
        "memory_usage": "3-6 GB",
    },
    "custom": {
        "name": "Custom Configuration",
        "description": "Configure all options manually",
        "icon": "⚙️",
        "analysis": {
            "quality_analysis": True,
            "diversity_analysis": True,
            "taxonomy_analysis": True,
            "novelty_detection": True,
            "max_sequences": 0,
        },
        "model": {
            "mode": "train_new",
            "architecture": "contrastive",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "scaling": {"enabled": True, "auto_configure": True},
        "estimated_time": "Varies",
        "memory_usage": "Varies",
    },
}


def render_configuration():
    """Render unified configuration interface"""

    # Show dataset info
    dataset = st.session_state.workflow_dataset
    if dataset:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(
                f"📂 Dataset: **{dataset.get('name')}** ({dataset.get('size_mb', 0):.2f} MB)"
            )
        with col2:
            if st.button("← Change Dataset"):
                st.session_state.workflow_step = 1
                st.rerun()

    st.divider()

    # Preset selection
    st.markdown("### Configuration Preset")

    preset_options = [
        f"{preset['icon']} {preset['name']}" for preset in PRESETS.values()
    ]

    current_preset = st.session_state.workflow_config.get("preset", "full_edna")
    current_index = list(PRESETS.keys()).index(current_preset)

    selected_preset_display = st.selectbox(
        "Choose a preset configuration",
        options=preset_options,
        index=current_index,
        key="workflow_preset_select",
    )

    # Extract preset key from selection
    selected_preset = list(PRESETS.keys())[
        preset_options.index(selected_preset_display)
    ]
    preset_config = PRESETS[selected_preset]

    # Show preset details
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Runtime", preset_config["estimated_time"])
    with col2:
        st.metric("Memory Usage", preset_config["memory_usage"])

    st.info(f"ℹ️ {preset_config['description']}")

    # Preset details expander
    with st.expander("📋 View Preset Details"):
        st.markdown("**Included Components:**")

        # Analysis components
        analysis_config = preset_config["analysis"]
        components = []
        if analysis_config.get("quality_analysis"):
            components.append("✓ Quality analysis with filtering")
        if analysis_config.get("diversity_analysis"):
            components.append("✓ Diversity metrics (alpha, beta, rarefaction)")
        if analysis_config.get("taxonomy_analysis"):
            components.append("✓ Taxonomy classification with BLAST")
        if analysis_config.get("novelty_detection"):
            components.append("✓ Novelty detection")

        # Model components
        model_config = preset_config["model"]
        if model_config.get("mode") == "train_new":
            epochs = model_config.get("epochs", 50)
            components.append(f"✓ Model training ({epochs} epochs)")
        elif model_config.get("mode") == "use_existing":
            components.append("✓ Use existing model")

        # Scaling components
        scaling_config = preset_config["scaling"]
        if scaling_config.get("enabled"):
            components.append("✓ Dynamic scaling with auto-configuration")

        for component in components:
            st.markdown(component)

        # Max sequences
        max_seq = analysis_config.get("max_sequences", 0)
        if max_seq > 0:
            st.markdown(f"\n**Sequence Limit:** {max_seq:,} sequences")
        else:
            st.markdown("\n**Sequence Limit:** All sequences (no limit)")

    st.divider()

    # Advanced options (collapsible)
    show_advanced = selected_preset == "custom" or st.checkbox(
        "Show Advanced Options", value=selected_preset == "custom"
    )

    if show_advanced:
        render_advanced_options(preset_config)

    # Save configuration
    st.session_state.workflow_config = {
        "preset": selected_preset,
        "analysis_settings": preset_config["analysis"].copy(),
        "model_settings": preset_config["model"].copy(),
        "scaling_settings": preset_config["scaling"].copy(),
    }

    # Navigation
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Dataset", use_container_width=True):
            st.session_state.workflow_step = 1
            st.rerun()

    with col2:
        if st.button("Start Execution →", type="primary", use_container_width=True):
            st.session_state.workflow_step = 3
            st.rerun()


def render_advanced_options(preset_config: Dict[str, Any]):
    """Render advanced configuration options"""

    st.markdown("### Advanced Options")

    # Analysis settings
    with st.expander("📊 Analysis Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            max_sequences = st.number_input(
                "Max Sequences (0 = all)",
                min_value=0,
                max_value=1000000,
                value=preset_config["analysis"].get("max_sequences", 0),
                step=1000,
                help="Limit number of sequences to process. Use 0 for no limit.",
            )

            quality_threshold = st.slider(
                "Quality Threshold",
                min_value=0,
                max_value=40,
                value=20,
                help="Minimum quality score for sequence filtering",
            )

        with col2:
            quality_analysis = st.checkbox(
                "Quality Analysis",
                value=preset_config["analysis"].get("quality_analysis", True),
            )

            diversity_analysis = st.checkbox(
                "Diversity Analysis",
                value=preset_config["analysis"].get("diversity_analysis", True),
            )

            taxonomy_analysis = st.checkbox(
                "Taxonomy Classification",
                value=preset_config["analysis"].get("taxonomy_analysis", True),
            )

            novelty_detection = st.checkbox(
                "Novelty Detection",
                value=preset_config["analysis"].get("novelty_detection", True),
            )

        # Update analysis settings
        st.session_state.workflow_config["analysis_settings"].update(
            {
                "max_sequences": max_sequences,
                "quality_threshold": quality_threshold,
                "quality_analysis": quality_analysis,
                "diversity_analysis": diversity_analysis,
                "taxonomy_analysis": taxonomy_analysis,
                "novelty_detection": novelty_detection,
            }
        )

    # Model settings
    with st.expander("🤖 Model Settings"):
        model_mode = st.radio(
            "Model Mode",
            options=["none", "use_existing", "train_new"],
            format_func=lambda x: {
                "none": "No Model",
                "use_existing": "Use Existing Model",
                "train_new": "Train New Model",
            }[x],
            index=["none", "use_existing", "train_new"].index(
                preset_config["model"].get("mode", "train_new")
            ),
        )

        if model_mode == "train_new":
            col1, col2 = st.columns(2)

            with col1:
                epochs = st.number_input(
                    "Training Epochs",
                    min_value=1,
                    max_value=500,
                    value=preset_config["model"].get("epochs", 50),
                )

                batch_size = st.selectbox(
                    "Batch Size",
                    options=[16, 32, 64, 128],
                    index=[16, 32, 64, 128].index(
                        preset_config["model"].get("batch_size", 32)
                    ),
                )

            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=preset_config["model"].get("learning_rate", 0.001),
                    format_func=lambda x: f"{x:.4f}",
                )

                architecture = st.selectbox(
                    "Architecture", options=["contrastive", "supervised"], index=0
                )

            st.session_state.workflow_config["model_settings"].update(
                {
                    "mode": model_mode,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "architecture": architecture,
                }
            )
        else:
            st.session_state.workflow_config["model_settings"]["mode"] = model_mode

    # Dynamic scaling settings
    with st.expander("⚡ Dynamic Scaling"):
        scaling_enabled = st.checkbox(
            "Enable Dynamic Scaling",
            value=preset_config["scaling"].get("enabled", True),
        )

        if scaling_enabled:
            auto_configure = st.checkbox(
                "Auto-configure (Recommended)",
                value=preset_config["scaling"].get("auto_configure", True),
                help="Automatically determine optimal scaling parameters",
            )

            if not auto_configure:
                col1, col2 = st.columns(2)

                with col1:
                    min_clusters = st.number_input(
                        "Min Clusters",
                        min_value=2,
                        max_value=1000,
                        value=preset_config["scaling"].get("min_clusters", 10),
                    )

                with col2:
                    max_clusters = st.number_input(
                        "Max Clusters",
                        min_value=10,
                        max_value=100000,
                        value=preset_config["scaling"].get("max_clusters", 10000),
                    )

                st.session_state.workflow_config["scaling_settings"].update(
                    {
                        "enabled": True,
                        "auto_configure": False,
                        "min_clusters": min_clusters,
                        "max_clusters": max_clusters,
                    }
                )
            else:
                st.session_state.workflow_config["scaling_settings"] = {
                    "enabled": True,
                    "auto_configure": True,
                }
        else:
            st.session_state.workflow_config["scaling_settings"]["enabled"] = False

    # Save as custom preset
    if st.button("💾 Save as Custom Preset"):
        st.success("Custom preset saved! (Feature to be implemented)")
