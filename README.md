# AI Pipeline Resilience Dashboard

A monitoring system that detects when AI models start behaving differently and automatically restores them to a working state.

## What This Does

Healthcare AI systems can degrade over time when the data they receive changes from what they were trained on. This is called "drift." This dashboard:

1. Monitors AI system behavior in real-time
2. Alerts you when something goes wrong (Warning, Critical, Emergency levels)
3. Automatically restores the system to a previous working configuration

## Quick Start

### Requirements

- Python 3.9 or higher
- PostgreSQL 12 or higher (or SQLite for testing)

### Installation

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python run_api.py
```

Then open http://localhost:8000 in your browser.

## Features

### Drift Detection

The system monitors three types of changes:

| Metric | What It Measures |
|--------|------------------|
| PSI (Population Stability Index) | Changes in input patterns |
| KS Test (Kolmogorov-Smirnov) | Changes in output quality |
| JS Divergence (Jensen-Shannon) | Changes in how the AI understands queries |

### Alert Levels

| Level | Meaning | Action |
|-------|---------|--------|
| Warning | Something may be changing | Monitor closely |
| Critical | Performance is degrading | Take action soon |
| Emergency | System needs immediate attention | Automatic rollback triggered |

### Rollback System

- Save snapshots of working configurations
- Restore to any previous version with one click
- Automatic restoration when critical issues are detected

## Importing Your Own Data

You can import your own query data from CSV or JSON files.

### Required Fields

- `query` - The text of the query
- `timestamp` - When the query occurred

### Optional Fields

- `query_category` - Category (e.g., appointment, billing)
- `confidence_score` - AI confidence (0.0 to 1.0)
- `ai_response` - The AI response text

### Import Command

```bash
python scripts/import_custom_data.py --file your_data.csv --format csv
```

See `data/examples/query_log_template.csv` for the expected format.

## Project Structure

```
backend/           API and business logic
frontend/          Dashboard interface
scripts/           Data generation and import tools
run_api.py         Start the API server
run_dashboard.py   Start the dashboard
```

## API Reference

| Endpoint | Description |
|----------|-------------|
| GET /api/drift/metrics | Current drift measurements |
| GET /api/drift/alerts | Active alerts |
| GET /api/rollback/versions | Saved configurations |
| POST /api/rollback/execute | Restore a configuration |

Full API documentation available at `/docs` when the server is running.

## Configuration

Edit `backend/config.py` to adjust alert thresholds:

```python
PSI_WARNING_THRESHOLD = 0.15
PSI_CRITICAL_THRESHOLD = 0.25
PSI_EMERGENCY_THRESHOLD = 0.40
DRIFT_WINDOW_MINUTES = 15
```

## Resetting Data

```bash
rm ai_resilience.db
python scripts/generate_sample_data.py
```

## License

MIT
