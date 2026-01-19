"""Script to run the Streamlit dashboard"""
import os
import subprocess
import sys

if __name__ == "__main__":
    port = os.getenv("PORT", "8501")
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/dashboard.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ])
