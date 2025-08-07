#!/usr/bin/env python3
"""
RTX 5090 Optimization Experiments Launcher
==========================================

Automated launcher that:
1. Checks system compatibility
2. Runs optimization experiments
3. Generates final report

Usage: python run_experiments.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_files_exist() -> bool:
    """Check if required files exist"""
    required_files = [
        "experiment_pipeline.py",
        "llm.py", 
        "check_system.py",
        "experiment_config.json",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Main launcher function"""
    print("🚀 RTX 5090 LLM Training Optimization Pipeline")
    print("=" * 50)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if files exist
    if not check_files_exist():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        return 1
    
    # Step 1: System compatibility check
    print("\n" + "="*50)
    print("STEP 1: System Compatibility Check")
    print("="*50)
    
    if not run_command("python check_system.py", "Running system compatibility check"):
        print("\n⚠️ System check completed with warnings.")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("Aborted by user.")
            return 1
    
    # Step 2: Run optimization experiments
    print("\n" + "="*50)
    print("STEP 2: Optimization Experiments")
    print("="*50)
    
    start_time = time.time()
    
    if not run_command("python experiment_pipeline.py", "Running optimization experiments"):
        print("\n❌ Experiments failed. Check the logs for details.")
        return 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Step 3: Generate visualizations
    print("\n" + "="*50)
    print("STEP 3: Generate Visualizations")
    print("="*50)
    
    run_command("python visualize_results.py", "Generating result visualizations")
    
    # Step 4: Display results
    print("\n" + "="*50)
    print("STEP 4: Results Summary")
    print("="*50)
    
    results_dir = Path("experiment_results")
    if results_dir.exists():
        print(f"📁 Results directory: {results_dir.absolute()}")
        
        # List generated files
        files = list(results_dir.glob("*"))
        if files:
            print("📄 Generated files:")
            for file in sorted(files):
                size_kb = file.stat().st_size / 1024
                print(f"   {file.name} ({size_kb:.1f} KB)")
        
        # Show report if it exists
        report_file = results_dir / "experiment_report.md"
        if report_file.exists():
            print(f"\n📊 Experiment report preview:")
            print("-" * 40)
            with open(report_file, 'r') as f:
                lines = f.readlines()
                # Show first 20 lines of the report
                for i, line in enumerate(lines[:20]):
                    print(f"   {line.rstrip()}")
                if len(lines) > 20:
                    print(f"   ... ({len(lines) - 20} more lines)")
            print("-" * 40)
    else:
        print("⚠️ Results directory not found")
    
    # Final summary
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"⏱️ Total execution time: {total_time/60:.1f} minutes")
    print(f"📊 View full report: experiment_results/experiment_report.md")
    print(f"📈 Raw data: experiment_results/experiment_results.json")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)