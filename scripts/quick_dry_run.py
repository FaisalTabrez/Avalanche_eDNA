#!/usr/bin/env python3
"""
Quick Pipeline Dry Run - Fast validation without heavy imports
"""

import sys
import os
from pathlib import Path
import subprocess

# Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def print_result(test_name, passed, message=""):
    if passed:
        print(f"{GREEN}[PASS]{RESET} {test_name}")
        if message:
            print(f"  {message}")
    else:
        print(f"{RED}[FAIL]{RESET} {test_name}")
        if message:
            print(f"  Error: {message}")

def print_warning(test_name, message):
    print(f"{YELLOW}[WARN]{RESET} {test_name}")
    print(f"  {message}")

print(f"\n{BLUE}{'='*70}")
print(f"  AVALANCHE eDNA - QUICK PIPELINE DRY RUN")
print(f"{'='*70}{RESET}\n")

errors = []
warnings = []
passed = []

# 1. Check Python version
print_header("1. Python Environment")
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print_result("Python version", True, f"v{py_version}")
passed.append("Python version")

# 2. Check critical files
print_header("2. Critical Files")
critical_files = [
    "scripts/run_pipeline.py",
    "streamlit_app.py",
    "config/config.yaml",
    "requirements.txt",
]

for file_path in critical_files:
    exists = Path(file_path).exists()
    print_result(file_path, exists)
    if exists:
        passed.append(file_path)
    else:
        errors.append(file_path)

# 3. Check directories
print_header("3. Directory Structure")
critical_dirs = [
    "src/database",
    "src/analysis",
    "src/security",
    "src/utils",
    "scripts",
    "tests",
    "data",
]

for dir_path in critical_dirs:
    exists = Path(dir_path).exists()
    print_result(dir_path, exists)
    if exists:
        passed.append(dir_path)
    else:
        errors.append(dir_path)

# 4. Compile all Python files
print_header("4. Python Syntax Validation")
result = subprocess.run(
    "find src scripts tests -name '*.py' -type f | wc -l",
    shell=True,
    capture_output=True,
    text=True
)
py_count = result.stdout.strip()
print(f"Found {py_count} Python files")

result = subprocess.run(
    "find src scripts tests -name '*.py' -type f -exec python -m py_compile {} \\; 2>&1 | grep -E 'Error|Traceback|SyntaxError' | wc -l",
    shell=True,
    capture_output=True,
    text=True
)
error_output = result.stdout.strip()
error_count = int(error_output) if error_output.isdigit() else 0
if error_count == 0:
    print_result("Python syntax check", True, f"All {py_count} files compile successfully")
    passed.append("Syntax validation")
else:
    print_result("Python syntax check", False, f"{error_count} files have errors")
    errors.append("Syntax validation")

# 5. Check Docker services
print_header("5. Docker Services")
services = ['redis', 'postgres', 'prometheus', 'grafana']
for service in services:
    result = subprocess.run(
        f"docker ps --filter 'name={service}' --format '{{{{.Status}}}}'",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and 'Up' in result.stdout:
        print_result(f"Docker: {service}", True, "Running")
        passed.append(f"Docker: {service}")
    else:
        print_warning(f"Docker: {service}", "Not running (optional)")
        warnings.append(f"Docker: {service}")

# 6. Check database files
print_header("6. Database Files")
db_file = Path("data/avalanche.db")
if db_file.exists():
    size_mb = db_file.stat().st_size / (1024 * 1024)
    print_result("SQLite database", True, f"Size: {size_mb:.2f} MB")
    passed.append("SQLite database")
else:
    print_warning("SQLite database", "Not found - will be created on first run")
    warnings.append("SQLite database")

# 7. Check reference data
print_header("7. Reference Data")
ref_dirs = [
    "reference/pr2",
    "reference/silva",
    "reference/eukref",
]
for ref_dir in ref_dirs:
    path = Path(ref_dir)
    if path.exists():
        file_count = len(list(path.glob("**/*")))
        print_result(ref_dir, True, f"{file_count} files")
        passed.append(ref_dir)
    else:
        print_warning(ref_dir, "Not found - download required")
        warnings.append(ref_dir)

# 8. Check configuration
print_header("8. Configuration")
if Path("config/config.yaml").exists():
    print_result("Config file", True)
    passed.append("Config file")
else:
    print_result("Config file", False, "config/config.yaml not found")
    errors.append("Config file")

if Path(".env.example").exists():
    print_result("Environment template", True, ".env.example found")
    passed.append("Environment template")
    
    if not Path(".env").exists():
        print_warning("Environment file", ".env not found - copy from .env.example")
        warnings.append("Environment file")

# Summary
print_header("VALIDATION SUMMARY")

total_tests = len(passed) + len(errors)
print(f"{GREEN}Passed:{RESET} {len(passed)}/{total_tests}")
print(f"{RED}Failed:{RESET} {len(errors)}/{total_tests}")
print(f"{YELLOW}Warnings:{RESET} {len(warnings)}")

if errors:
    print(f"\n{RED}Critical Errors:{RESET}")
    for error in errors:
        print(f"  • {error}")

if warnings:
    print(f"\n{YELLOW}Warnings (non-critical):{RESET}")
    for warning in warnings[:10]:  # Show first 10
        print(f"  • {warning}")
    if len(warnings) > 10:
        print(f"  ... and {len(warnings) - 10} more")

success_rate = (len(passed) / total_tests * 100) if total_tests > 0 else 0

print(f"\n{'='*70}")
if success_rate >= 90 and len(errors) == 0:
    print(f"{GREEN}✓ PIPELINE READY - {success_rate:.1f}% validation passed{RESET}")
    print(f"{GREEN}  You can proceed with running the pipeline{RESET}")
    sys.exit(0)
elif success_rate >= 70:
    print(f"{YELLOW}⚠ PIPELINE PARTIALLY READY - {success_rate:.1f}% validation passed{RESET}")
    print(f"{YELLOW}  Review warnings before running the pipeline{RESET}")
    sys.exit(0)
else:
    print(f"{RED}✗ PIPELINE NOT READY - {success_rate:.1f}% validation passed{RESET}")
    print(f"{RED}  Fix critical errors before running the pipeline{RESET}")
    sys.exit(1)
