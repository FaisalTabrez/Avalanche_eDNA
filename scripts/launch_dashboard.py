#!/usr/bin/env python
"""
Launch script for the eDNA Biodiversity Assessment Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    # Get the path to the dashboard script
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / "src"
    dashboard_script = src_dir / "visualization" / "dashboard.py"
    
    # Ensure the dashboard script exists
    if not dashboard_script.exists():
        print(f"Error: Dashboard script not found at {dashboard_script}")
        sys.exit(1)
    
    # Add src to Python path
    sys.path.insert(0, str(src_dir))
    
    # Set environment variables for the dashboard
    os.environ['PYTHONPATH'] = str(src_dir)
    
    print("🌊 Launching eDNA Biodiversity Assessment Dashboard...")
    print(f"📁 Dashboard script: {dashboard_script}")
    print("🌐 The dashboard will be available at: http://localhost:8501")
    print("📝 Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_script),
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()