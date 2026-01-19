"""Main FastAPI application"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import drift, rollback
from backend.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    init_db()
    yield

app = FastAPI(
    title="AI Pipeline Resilience API",
    description="Drift Detection and Rollback System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(drift.router)
app.include_router(rollback.router)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "AI Pipeline Resilience API",
        "version": "1.0.0",
        "endpoints": {
            "drift": "/api/drift",
            "rollback": "/api/rollback"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
