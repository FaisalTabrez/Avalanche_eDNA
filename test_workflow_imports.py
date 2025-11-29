"""
Test script to check for import errors in workflow components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")
print("=" * 60)

# Test 1: Task Manager
try:
    from src.ui.task_manager import get_task_manager, TaskManager, TaskStatus, TaskType, TaskInfo
    print("✓ Task Manager imports successful")
    
    # Test instantiation
    tm = get_task_manager()
    print(f"  - TaskManager instance created: {type(tm).__name__}")
    print(f"  - State dir: {tm.state_dir}")
except Exception as e:
    print(f"✗ Task Manager import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Workflow Hub
try:
    from src.ui.workflow.workflow_hub import render_workflow_hub, init_workflow_state
    print("✓ Workflow Hub imports successful")
except Exception as e:
    print(f"✗ Workflow Hub import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Step 1 - Dataset
try:
    from src.ui.workflow.step_1_dataset import render_dataset_selection
    print("✓ Step 1 (Dataset) imports successful")
except Exception as e:
    print(f"✗ Step 1 import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Step 2 - Configure
try:
    from src.ui.workflow.step_2_configure import render_configuration, PRESETS
    print("✓ Step 2 (Configure) imports successful")
    print(f"  - Available presets: {list(PRESETS.keys())}")
except Exception as e:
    print(f"✗ Step 2 import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 5: Step 3 - Execute
try:
    from src.ui.workflow.step_3_execute import render_execution, start_pipeline_execution
    print("✓ Step 3 (Execute) imports successful")
except Exception as e:
    print(f"✗ Step 3 import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Step 4 - Results
try:
    from src.ui.workflow.step_4_results import render_results
    print("✓ Step 4 (Results) imports successful")
except Exception as e:
    print(f"✗ Step 4 import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 7: Workflow package
try:
    from src.ui.workflow import render_workflow_hub as rwh
    print("✓ Workflow package import successful")
except Exception as e:
    print(f"✗ Workflow package import failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Import testing complete!")
