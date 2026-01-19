"""
Utility script to refresh recent queries to ensure there are always queries
in the active window. Can be run periodically (e.g., via cron/scheduler) or
manually when needed.

Usage:
    python scripts/refresh_recent_queries.py
    
For scheduling (every 10 minutes):
    # Windows Task Scheduler or cron job
    python scripts/refresh_recent_queries.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, init_db
from backend.models.query_log import QueryLog

# Import from generate_sample_data
from scripts.generate_sample_data import refresh_recent_queries

def main():
    """Refresh recent queries to keep active window populated"""
    init_db()
    db = SessionLocal()
    
    try:
        # Check how many queries are in the last 15 minutes
        cutoff = datetime.now() - timedelta(minutes=15)
        recent_count = db.query(QueryLog).filter(QueryLog.timestamp >= cutoff).count()
        
        # Add queries to ensure we have at least 1500 in the active window
        target_count = 1500
        if recent_count < target_count:
            needed = target_count - recent_count
            refresh_recent_queries(db, n_samples=needed + 200)  # Add buffer
    
    finally:
        db.close()

if __name__ == "__main__":
    main()
