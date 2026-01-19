"""Configuration settings for the application"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def get_database_url() -> str:
    """Get database URL, defaulting to SQLite for easy setup"""
    # Default to SQLite for easy local development
    default_url = f"sqlite:///{PROJECT_ROOT}/ai_resilience.db"
    url = os.getenv("DATABASE_URL", default_url)
    # Some providers use postgres:// but SQLAlchemy requires postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database (PostgreSQL by default, SQLite for development)
    DATABASE_URL: str = get_database_url()
    
    # Drift Detection Thresholds
    PSI_WARNING_THRESHOLD: float = 0.15
    PSI_CRITICAL_THRESHOLD: float = 0.25
    PSI_EMERGENCY_THRESHOLD: float = 0.40
    
    KS_WARNING_THRESHOLD: float = 0.05  # p-value threshold
    KS_CRITICAL_THRESHOLD: float = 0.01
    KS_EMERGENCY_THRESHOLD: float = 0.001
    
    JS_WARNING_THRESHOLD: float = 0.1
    JS_CRITICAL_THRESHOLD: float = 0.2
    JS_EMERGENCY_THRESHOLD: float = 0.3

    # Wasserstein Distance thresholds (for embedding drift)
    WASSERSTEIN_WARNING_THRESHOLD: float = 0.5
    WASSERSTEIN_CRITICAL_THRESHOLD: float = 1.5
    WASSERSTEIN_EMERGENCY_THRESHOLD: float = 2.5

    # Rolling window
    DRIFT_WINDOW_MINUTES: int = 15
    
    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignore extra environment variables
    )

settings = Settings()
