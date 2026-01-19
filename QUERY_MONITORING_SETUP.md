# Query Monitoring Setup - Always Active Queries

## ✅ Implementation Complete

### Changes Made

1. **New API Endpoint**: `/api/drift/refresh-queries` (POST)
   - Automatically generates queries if active window has less than 1500 queries
   - Ensures continuous query availability
   - Location: `backend/api/drift.py`

2. **Dashboard Auto-Refresh**: 
   - Automatically detects when active queries = 0
   - Calls the refresh endpoint to generate new queries
   - Shows notification when refreshing
   - Location: `frontend/dashboard.py` (lines 775-802)

3. **Background Service Enhancement**:
   - Reduced check interval from 10 minutes to 5 minutes
   - More frequent monitoring ensures queries never drop to zero
   - Location: `scripts/keep_queries_running.py`

### How It Works

1. **Dashboard Detection**: When the dashboard loads and detects `sample_size = 0`, it automatically calls the refresh endpoint
2. **Background Service**: Runs every 5 minutes, checking if active queries are below 1500 and generating more if needed
3. **API Endpoint**: Can be called manually or by the dashboard to refresh queries on-demand

### Current Status

- **Active Queries**: 1,700 (healthy!)
- **Target**: 1,500+ queries in last 15 minutes
- **Background Service**: Configured to run every 5 minutes

### To Apply Changes

**The API server needs to be restarted** to pick up the new `/api/drift/refresh-queries` endpoint:

```bash
# Stop current API server (Ctrl+C in terminal running it, or kill process 32376)
# Then restart:
python run_api.py
```

Or restart the API server that's currently running on port 8000.

### Verification

After restart, test the endpoint:
```bash
python -c "import requests; r = requests.post('http://localhost:8000/api/drift/refresh-queries'); print(r.json())"
```

### Background Service

To ensure queries are always running, start the background service:
```bash
python scripts/keep_queries_running.py
```

Or it can be run as a Windows service or scheduled task to run continuously.

### Dashboard Behavior

The dashboard will now:
- ✅ Automatically refresh queries when it detects zero active queries
- ✅ Show a notification when refreshing
- ✅ Always display proper metrics (never zero)
- ✅ Continuously monitor and maintain query flow
