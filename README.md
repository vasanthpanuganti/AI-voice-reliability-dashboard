# AI Pipeline Resilience Dashboard

Monitors AI systems for behavioral drift and automatically restores them to working configurations.

## What This Does

1. **Monitors** AI system behavior in real-time
2. **Alerts** when performance degrades (Warning, Critical, Emergency)
3. **Automatically rolls back** to a previous working configuration

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows (or: source venv/bin/activate)
pip install -r requirements.txt
python scripts/generate_sample_data.py

# Run (two terminals)
python run_api.py        # Terminal 1: API on port 8000
python run_dashboard.py  # Terminal 2: Dashboard on port 8501
```

**Open:** http://localhost:8501

## Key Metrics

| Metric | Measures | Normal | Warning | Critical |
|--------|----------|--------|---------|----------|
| **PSI** | Input pattern changes | < 0.10 | 0.10-0.25 | > 0.25 |
| **KS p-value** | Output quality | > 0.05 | 0.01-0.05 | < 0.01 |
| **JS Divergence** | AI understanding | < 0.10 | 0.10-0.20 | > 0.20 |

## Features

- **Drift Detection** - Monitors input patterns, output quality, and AI understanding
- **Alert System** - Three severity levels with automated responses
- **Rollback Control** - Save and restore configuration snapshots
- **Segment Analysis** - Detect drift in specific query categories

## Project Structure

```
backend/           API and services
frontend/          Streamlit dashboard
scripts/           Data generation
run_api.py         API server (port 8000)
run_dashboard.py   Dashboard (port 8501)
```

## API

| Endpoint | Description |
|----------|-------------|
| GET /api/drift/metrics | Current drift metrics |
| GET /api/drift/alerts | Active alerts |
| GET /api/rollback/versions | Saved configurations |
| POST /api/rollback/execute | Restore a version |

**Docs:** http://localhost:8000/docs

## Configuration

Edit `backend/config.py`:

```python
PSI_WARNING_THRESHOLD = 0.15
PSI_CRITICAL_THRESHOLD = 0.25
DRIFT_WINDOW_MINUTES = 15
```

## Reset Data

```bash
rm ai_resilience.db && python scripts/generate_sample_data.py
```

## License

MIT
