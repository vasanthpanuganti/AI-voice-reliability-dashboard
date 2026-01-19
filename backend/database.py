"""Database connection and session management"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings
import os
from pathlib import Path

# Create database engine with appropriate settings
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite needs check_same_thread=False for FastAPI
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, echo=False, connect_args=connect_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db():
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def apply_migrations():
    """Apply pending database migrations"""
    inspector = inspect(engine)
    
    # Check if configurations table exists
    if 'configurations' in inspector.get_table_names():
        # Check if current_version_id column exists
        columns = [col['name'] for col in inspector.get_columns('configurations')]
        if 'current_version_id' not in columns:
            # Apply migration: add current_version_id column
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE configurations ADD COLUMN current_version_id INTEGER"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_configurations_current_version_id ON configurations(current_version_id)"))
                conn.commit()
                print("Applied migration: Added current_version_id to configurations table")

def init_db():
    """Initialize database tables"""
    # Import all models to register them with Base
    from backend.models import query_log, configuration, drift_metrics, rollback, baseline_statistics
    Base.metadata.create_all(bind=engine)
    
    # Apply any pending migrations
    apply_migrations()
