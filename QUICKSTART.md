# AI Pipeline Resilience Dashboard - Quick Start

## Prerequisites

- Python 3.9+
- PostgreSQL 12+ (or use SQLite for development)

## Setup (2 minutes)

```bash
# 1. Create PostgreSQL database
createdb ai_resilience_db
# Or via psql:
# CREATE DATABASE ai_resilience_db;

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure database (optional - PostgreSQL is default)
# Create .env file with:
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_resilience_db
#
# For SQLite development (optional):
# DATABASE_URL=sqlite:///./ai_resilience.db

# 5. Initialize database and generate sample data
python scripts/generate_sample_data.py
```

## Run the Dashboard

**Terminal 1 - Start API:**
```bash
python run_api.py
```

**Terminal 2 - Start Dashboard:**
```bash
python run_dashboard.py
```

**Open browser:** http://localhost:8501

## What You'll See

### Drift Detection Page
- **PSI Score**: Measures input distribution shift (0.08 = normal, >0.25 = critical)
- **KS Test p-value**: Detects output distribution changes (lower = more drift)
- **JS Divergence**: Measures embedding space shift

### Active Alerts
Each alert shows:
- What triggered it (metric value vs. threshold)
- Why it matters (explanation)
- What to do (recommended actions)

### Rollback Control Page
- Current configuration (embedding model, thresholds)
- Version history with performance metrics
- Automated rollback history (triggered by critical alerts)

## How It Works

1. **Sample Data**: The script generates 1,200 healthcare queries simulating:
   - 7 days of baseline data (normal distribution)
   - Recent data with 30% billing category shift

2. **Drift Detection**: The system compares current queries to baseline:
   - PSI measures category distribution changes
   - KS Test detects confidence score shifts
   - JS Divergence tracks embedding space drift

3. **Automated Rollback**: When critical/emergency alerts trigger:
   - System automatically restores the last known-good configuration
   - All changes are logged for audit

## API Documentation

Visit http://localhost:8000/docs for interactive API docs.

## Reset Data

### PostgreSQL
```bash
# Drop and recreate database
dropdb ai_resilience_db
createdb ai_resilience_db
python scripts/generate_sample_data.py
```

### SQLite (Development)
```bash
# Delete database file and regenerate
rm ai_resilience.db
python scripts/generate_sample_data.py
```
