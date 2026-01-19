"""Vercel entrypoint for FastAPI application"""
from backend.api.main import app

# Vercel expects the app variable to be available
__all__ = ["app"]
