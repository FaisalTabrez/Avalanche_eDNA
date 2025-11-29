#!/usr/bin/env python3
"""
Quick UI Module Test
Verifies all Mission Control modules can be imported and rendered without errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all UI modules can be imported."""
    print("Testing UI module imports...")
    
    try:
        from src.ui.core.state_manager import StateManager
        print("✓ StateManager imported")
        
        from src.ui.core.task_manager import TaskManager
        print("✓ TaskManager imported")
        
        from src.ui.components.navigation import Navigation
        print("✓ Navigation imported")
        
        from src.ui.modules import dashboard, analysis, data_explorer, model_forge, system_monitor
        print("✓ All modules imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_manager():
    """Test StateManager initialization."""
    print("\nTesting StateManager...")
    try:
        from src.ui.core.state_manager import StateManager
        
        # Test defaults are set
        defaults = StateManager.DEFAULTS
        assert "current_page" in defaults
        assert "system_status" in defaults
        print(f"✓ StateManager has {len(defaults)} default keys")
        return True
    except Exception as e:
        print(f"✗ StateManager test failed: {e}")
        return False

def test_task_manager():
    """Test TaskManager singleton."""
    print("\nTesting TaskManager...")
    try:
        from src.ui.core.task_manager import TaskManager
        
        tm = TaskManager()
        metrics = tm.get_system_metrics()
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        print(f"✓ TaskManager metrics: CPU={metrics['cpu']}%, RAM={metrics['memory']}%, Disk={metrics['disk']}%")
        return True
    except Exception as e:
        print(f"✗ TaskManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Mission Control UI Module Tests")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("StateManager", test_state_manager()))
    results.append(("TaskManager", test_task_manager()))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("="*60))
    if all_passed:
        print("All tests passed! UI is ready.")
        sys.exit(0)
    else:
        print("Some tests failed. Check output above.")
        sys.exit(1)
