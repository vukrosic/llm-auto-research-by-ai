#!/usr/bin/env python3
"""
Dashboard Launcher for RTX 5090 Optimization Experiments
========================================================

Launches the Streamlit dashboard with proper configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit_installed():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def check_results_exist():
    """Check if experiment results exist"""
    results_file = Path("experiment_results/experiment_results.json")
    return results_file.exists()

def main():
    """Launch the Streamlit dashboard"""
    print("🚀 RTX 5090 Optimization Dashboard Launcher")
    print("=" * 50)
    
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        print("❌ Streamlit not installed.")
        print("Install with: pip install streamlit plotly pandas")
        print("Or: pip install -r requirements.txt")
        return 1
    
    # Check if results exist
    if not check_results_exist():
        print("⚠️ No experiment results found.")
        print("The dashboard will show instructions to run experiments first.")
        print()
        response = input("Launch dashboard anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("Run experiments first with: python run_experiments.py")
            return 1
    else:
        print("✅ Experiment results found.")
    
    # Launch Streamlit
    print("\n🌐 Launching Streamlit dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use the sidebar to filter results")
    print("   - Explore different tabs for various analyses")
    print("   - Hover over charts for detailed information")
    print("\n⏹️ Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n⏹️ Dashboard stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching dashboard: {e}")
        return 1
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install with: pip install streamlit")
        return 1

if __name__ == "__main__":
    sys.exit(main())