"""
Background service to continuously keep queries in the active window.
This ensures the dashboard always shows active queries.

Run this in a separate terminal or as a background process:
    python scripts/keep_queries_running.py

The script will:
- Check every 10 minutes if there are enough queries in the active window
- Automatically add new queries if needed
- Run silently without output
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, init_db
from backend.models.query_log import QueryLog
from scripts.generate_sample_data import refresh_recent_queries

def check_and_refresh():
    """Check if queries need refreshing and add them if needed"""
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
            refresh_recent_queries(db, n_samples=needed + 300)  # Add buffer
    finally:
        db.close()

def main():
    """Run continuously, checking every 10 minutes"""
    while True:
        try:
            check_and_refresh()
        except KeyboardInterrupt:
            # Allow clean shutdown on Ctrl+C
            break
        except Exception as e:
            # Log error but keep running (silent mode for production)
            # Uncomment for debugging:
            # print(f"Error refreshing queries: {e}")
            pass
        
        # Sleep for 5 minutes (check more frequently to ensure queries always exist)
        time.sleep(300)

if __name__ == "__main__":
    main()
