# AI Pipeline Resilience Dashboard

Drift Detection and Rollback System for Healthcare AI Pipelines

## Problem Statement

Healthcare AI systems need to detect when input data distributions shift from training baselines (drift) and automatically recover by rolling back to known-good configurations. This dashboard demonstrates:

1. **Drift Detection** - Monitor input, output, and embedding space shifts using statistical methods
2. **Automated Alerts** - Generate alerts when thresholds are breached (Warning → Critical → Emergency)
3. **Automated Rollback** - Automatically restore previous known-good configurations when critical drift is detected

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 12+ (or use SQLite for development)

### Setup

```bash
# 1. Create PostgreSQL database
createdb ai_resilience_db
# Or via psql:
# CREATE DATABASE ai_resilience_db;

# 2. Set environment variable (optional - PostgreSQL is default)
# Create .env file with:
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_resilience_db
# 
# For SQLite development (optional):
# DATABASE_URL=sqlite:///./ai_resilience.db

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database and generate sample data
python scripts/generate_sample_data.py

# 5. Start the API (Terminal 1)
python run_api.py

# 6. Start the Dashboard (Terminal 2)
python run_dashboard.py

# 7. Open http://localhost:8501
```

## Features

### Drift Detection
- **PSI (Population Stability Index)** - Measures input distribution shift
- **KS Test (Kolmogorov-Smirnov)** - Detects sharp changes in output distributions
- **JS Divergence (Jensen-Shannon)** - Measures embedding space drift
- Real-time monitoring with 15-minute rolling windows

### Alerting System
| Severity | PSI Threshold | KS p-value | JS Divergence |
|----------|--------------|------------|---------------|
| Warning  | > 0.15       | < 0.05     | > 0.1         |
| Critical | > 0.25       | < 0.01     | > 0.2         |
| Emergency| > 0.40       | < 0.001    | > 0.3         |

Each alert includes:
- What the metric means and why it triggered
- Baseline vs. current period comparison
- Category distribution shifts
- Recommended actions

### Rollback System
- **Configuration Versioning** - Snapshot and track all configuration changes
- **Known-Good Versions** - Mark versions that performed well
- **Manual Rollback** - One-click restore to any previous version
- **Automated Rollback** - Auto-restore when critical/emergency alerts trigger
- **Atomic Operations** - All-or-nothing restoration with audit logging

## Sample Data

The demo generates realistic healthcare query data:

| Category | Baseline Distribution | Drifted Distribution |
|----------|----------------------|---------------------|
| Appointment | 35% | 15% |
| Prescription | 25% | 15% |
| Billing | 20% | **50%** (major shift!) |
| Clinical | 10% | 10% |
| General | 10% | 10% |

This simulates a scenario where billing queries suddenly increased from 20% to 50% - triggering drift alerts.

## Project Structure

```
ai-resilience-dashboard/
├── backend/
│   ├── api/              # FastAPI endpoints
│   ├── models/           # SQLAlchemy models (PostgreSQL/SQLite)
│   ├── services/         # Drift detection & rollback logic
│   └── utils/            # Statistical functions (PSI, KS, JS)
├── frontend/
│   └── dashboard.py      # Streamlit UI
├── scripts/
│   └── generate_sample_data.py  # Demo data generator
├── run_api.py            # API server (port 8000)
└── run_dashboard.py      # Dashboard server (port 8501)
```

## API Endpoints

### Drift Detection
- `GET /api/drift/metrics` - Current drift metrics
- `GET /api/drift/alerts` - Active alerts
- `GET /api/drift/history` - Historical drift data
- `GET /api/drift/alerts/{id}/diagnostics` - Alert explanation

### Rollback
- `GET /api/rollback/versions` - Configuration versions
- `GET /api/rollback/current-config` - Current configuration
- `POST /api/rollback/execute` - Execute rollback
- `GET /api/rollback/history` - Rollback audit log

## Dashboard Pages

1. **Drift Detection** - Key metrics, active alerts with explanations, historical trends
2. **Rollback Control** - Current config, version history, rollback execution
3. **System Overview** - Summary statistics and health metrics

## Technical Details

- **Database**: PostgreSQL (default) or SQLite (for development)
- **API**: FastAPI with automatic OpenAPI docs at `/docs`
- **Frontend**: Streamlit for interactive dashboard
- **Statistical Methods**: PSI, KS Test, Jensen-Shannon Divergence

### Database Configuration

The system uses PostgreSQL by default. To use SQLite for local development, set:
```bash
DATABASE_URL=sqlite:///./ai_resilience.db
```

Default PostgreSQL connection (configured in `backend/config.py`):
```
postgresql://postgres:postgres@localhost:5432/ai_resilience_db
```

## Regenerating Data

To reset and regenerate sample data:

```bash
# Delete existing database
rm ai_resilience.db

# Generate fresh sample data
python scripts/generate_sample_data.py
```

## Configuration

Key thresholds in `backend/config.py`:

```python
# PSI Thresholds
PSI_WARNING_THRESHOLD = 0.15
PSI_CRITICAL_THRESHOLD = 0.25
PSI_EMERGENCY_THRESHOLD = 0.40

# Rolling window for drift detection
DRIFT_WINDOW_MINUTES = 15
```

## License

MIT
