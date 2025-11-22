#!/usr/bin/env python3
"""
Quick startup script for development
Runs startup checks and launches Streamlit with optimizations
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Run startup checks
from scripts.startup import startup_checks

def main():
    """Main entry point"""
    print("\n🚀 Starting Avalanche eDNA Platform...")
    
    # Run startup checks
    success = startup_checks()
    
    if success:
        print("\n✓ All systems ready!")
    else:
        print("\n⚠ Some components had issues - check logs above")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Startup cancelled.")
            sys.exit(1)
    
    # Import and run streamlit
    print("\n" + "=" * 80)
    print("Launching Streamlit Application...")
    print("=" * 80)
    print("\nAccess the app at: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    # Use streamlit CLI to run the app
    import subprocess
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.headless=true"
    ])


if __name__ == '__main__':
    main()
