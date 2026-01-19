"""Script to run test suite and generate coverage report"""
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run pytest with coverage and generate report"""
    project_root = Path(__file__).parent.parent
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=backend",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--tb=short",
        # "-x"  # Stop on first failure (uncomment for faster debugging)
    ]
    
    print("Running test suite...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above.")
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
