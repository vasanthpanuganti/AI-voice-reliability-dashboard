"""Vercel entrypoint for FastAPI application"""
import sys
from pathlib import Path

# Add project root to path for imports
root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.api.main import app

# Vercel requires 'app' to be directly accessible
# This is the FastAPI application instance
__all__ = ["app"]

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
