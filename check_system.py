#!/usr/bin/env python3
"""
System Compatibility Checker for RTX 5090 Optimization Pipeline
===============================================================

Checks hardware, software, and dependency requirements before running experiments.
"""

import torch
import sys
import subprocess
import importlib
import psutil
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_cuda_availability() -> Tuple[bool, str]:
    """Check CUDA availability and version"""
    if not torch.cuda.is_available():
        return False, "❌ CUDA not available"
    
    cuda_version = torch.version.cuda
    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    status = f"✅ CUDA {cuda_version}, {device_count} GPU(s)"
    status += f"\n   Primary GPU: {gpu_name} ({memory_gb:.1f}GB)"
    
    # Check if it's RTX 5090
    if "5090" in gpu_name:
        status += "\n   🎯 RTX 5090 detected - optimal configuration!"
    elif "RTX" in gpu_name or "GeForce" in gpu_name:
        status += "\n   ⚠️ Non-5090 GPU detected - may need config adjustments"
    else:
        status += "\n   ⚠️ Unknown GPU - performance may vary"
        
    return True, status

def check_memory_requirements() -> Tuple[bool, str]:
    """Check system memory requirements"""
    memory = psutil.virtual_memory()
    total_gb = memory.total / 1e9
    available_gb = memory.available / 1e9
    
    if total_gb >= 32:
        status = f"✅ System RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available"
    elif total_gb >= 16:
        status = f"⚠️ System RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available (32GB+ recommended)"
    else:
        status = f"❌ System RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available (insufficient)"
        return False, status
        
    return True, status

def check_required_packages() -> Tuple[bool, str]:
    """Check if required packages are installed"""
    required_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.30.0"),
        ("datasets", "2.12.0"),
        ("tqdm", "4.65.0"),
        ("numpy", "1.24.0"),
    ]
    
    optional_packages = [
        ("bitsandbytes", "0.41.0"),
        ("accelerate", "0.20.0"),
    ]
    
    results = []
    all_required_ok = True
    
    # Check required packages
    for package, min_version in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            results.append(f"✅ {package}: {version}")
        except ImportError:
            results.append(f"❌ {package}: not installed (required)")
            all_required_ok = False
    
    # Check optional packages
    for package, min_version in optional_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            results.append(f"✅ {package}: {version} (optional)")
        except ImportError:
            results.append(f"⚠️ {package}: not installed (optional, but recommended)")
    
    status = "\n".join(results)
    return all_required_ok, status

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    disk = psutil.disk_usage('.')
    free_gb = disk.free / 1e9
    total_gb = disk.total / 1e9
    
    if free_gb >= 10:
        return True, f"✅ Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB total"
    elif free_gb >= 5:
        return True, f"⚠️ Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB total (low)"
    else:
        return False, f"❌ Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB total (insufficient)"

def estimate_experiment_time() -> str:
    """Estimate total experiment time"""
    # Based on configuration
    num_configs = 4  # Small, Medium, Large, XLarge
    num_optimizations = 4  # Baseline, GQA, FP4, Both
    time_per_experiment = 30  # minutes
    
    total_experiments = num_configs * num_optimizations
    total_time_hours = (total_experiments * time_per_experiment) / 60
    
    return f"📊 Estimated experiment time: {total_experiments} experiments × {time_per_experiment}min = {total_time_hours:.1f} hours"

def generate_install_commands() -> List[str]:
    """Generate installation commands for missing dependencies"""
    commands = [
        "# Install PyTorch with CUDA support",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "",
        "# Install other required packages",
        "pip install transformers datasets tqdm numpy",
        "",
        "# Install optional but recommended packages",
        "pip install bitsandbytes accelerate",
        "",
        "# Or install all at once from requirements.txt",
        "pip install -r requirements.txt"
    ]
    return commands

def main():
    """Run all system checks"""
    print("🔍 RTX 5090 Optimization Pipeline - System Compatibility Check")
    print("=" * 65)
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA Support", check_cuda_availability),
        ("System Memory", check_memory_requirements),
        ("Required Packages", check_required_packages),
        ("Disk Space", check_disk_space),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            passed, message = check_func()
            print(f"  {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ❌ Error during check: {e}")
            all_passed = False
    
    print(f"\n{estimate_experiment_time()}")
    
    print("\n" + "=" * 65)
    
    if all_passed:
        print("🎉 All checks passed! System is ready for optimization experiments.")
        print("\nTo start experiments, run:")
        print("  python experiment_pipeline.py")
    else:
        print("⚠️ Some checks failed. Please address the issues above.")
        print("\nInstallation commands:")
        for cmd in generate_install_commands():
            print(f"  {cmd}")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()