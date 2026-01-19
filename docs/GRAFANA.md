# Grafana Integration Guide

This guide explains how to integrate the AI Pipeline Resilience Dashboard with Grafana using the JSON API datasource.

## Overview

The API provides time-series endpoints that are compatible with Grafana's JSON API datasource. You can visualize drift metrics, alerts, and rollback events in Grafana dashboards.

## Prerequisites

- Grafana instance (v8.0+)
- AI Pipeline Resilience API running (default: `http://localhost:8000`)
- JSON API datasource plugin (included in Grafana 8.0+)

## Step 1: Add JSON API Datasource

1. In Grafana, go to **Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **JSON API**
4. Configure:
   - **Name**: `AI Resilience API`
   - **URL**: `http://localhost:8000`
   - **HTTP Method**: `GET`
   - Click **Save & Test**

## Step 2: Configure Endpoints

The API exposes the following time-series compatible endpoints:

### `/api/drift/history`

Returns historical drift metrics with timestamps.

**Example Response:**
```json
[
  {
    "id": 1,
    "timestamp": "2026-01-18T10:00:00",
    "psi_score": 0.15,
    "ks_p_value": 0.03,
    "js_divergence": 0.12,
    "sample_size": 1000
  }
]
```

**Grafana Query Configuration:**
- **Method**: GET
- **URL Path**: `/api/drift/history?limit=100`
- **Field**: `psi_score` (or `ks_p_value`, `js_divergence`)
- **Time Field**: `timestamp`

### `/api/drift/metrics`

Returns current drift metrics (single point in time).

**Example Response:**
```json
{
  "psi_score": 0.18,
  "ks_statistic": 0.25,
  "ks_p_value": 0.02,
  "js_divergence": 0.15,
  "timestamp": "2026-01-18T10:00:00",
  "sample_size": 1500
}
```

### `/api/drift/alerts`

Returns alert counts over time (can be aggregated by Grafana).

**Example Response:**
```json
[
  {
    "id": 1,
    "metric_type": "input_drift",
    "metric_name": "psi_score",
    "metric_value": 0.45,
    "severity": "emergency",
    "status": "active",
    "created_at": "2026-01-18T10:00:00"
  }
]
```

## Step 3: Create Panels

### Panel 1: PSI Score Over Time

1. Create a new panel
2. Select **Time series** visualization
3. Configure query:
   - **Data source**: `AI Resilience API`
   - **Method**: `GET`
   - **URL Path**: `/api/drift/history?limit=1000`
   - **Field**: `psi_score`
   - **Time Field**: `timestamp`
4. Add threshold lines:
   - Warning: `0.15` (orange)
   - Critical: `0.25` (red)
   - Emergency: `0.40` (dark red)

### Panel 2: JS Divergence Over Time

1. Create a new panel
2. Select **Time series** visualization
3. Configure query:
   - **Data source**: `AI Resilience API`
   - **Method**: `GET`
   - **URL Path**: `/api/drift/history?limit=1000`
   - **Field**: `js_divergence`
   - **Time Field**: `timestamp`

### Panel 3: Active Alert Count

1. Create a new panel
2. Select **Stat** visualization
3. Configure query:
   - **Data source**: `AI Resilience API`
   - **Method**: `GET`
   - **URL Path**: `/api/drift/alerts?status=active`
   - Use Grafana transformation: **Reduce** → **Count**

### Panel 4: Alert Severity Distribution

1. Create a new panel
2. Select **Pie chart** visualization
3. Configure query:
   - **Data source**: `AI Resilience API`
   - **Method**: `GET`
   - **URL Path**: `/api/drift/alerts?status=active`
   - Use Grafana transformation: **Group by** `severity`

## Step 4: Advanced Queries

### Filter by Time Range

Grafana automatically passes time range in the query, but our API accepts `limit` parameter:

```
/api/drift/history?limit=1000
```

You can modify the query to filter results based on Grafana's time range using transformations.

### Multiple Metrics in One Panel

To show PSI, KS, and JS in one panel:

1. Create 3 queries in the same panel
2. Each query points to `/api/drift/history?limit=1000`
3. Use different **Field** selections: `psi_score`, `ks_p_value`, `js_divergence`

## Example Dashboard JSON

A pre-configured dashboard JSON file is available at `grafana/dashboard.json` that you can import directly into Grafana.

To import:
1. In Grafana, go to **Dashboards** → **Import**
2. Upload or paste the contents of `grafana/dashboard.json`
3. Select the `AI Resilience API` datasource
4. Click **Import**

## Troubleshooting

### Connection Issues

- Verify API is running: `curl http://localhost:8000/health`
- Check API logs for errors
- Verify CORS settings if accessing from different domain

### Data Not Appearing

- Check that timestamps are in ISO format (our API returns ISO format)
- Verify the **Time Field** in Grafana matches `timestamp`
- Ensure the **Field** name matches the JSON key (e.g., `psi_score`)

### Performance

- Use `limit` parameter to control data volume: `/api/drift/history?limit=100`
- Adjust panel refresh intervals (30s recommended)
- Consider caching if querying frequently

## API Endpoint Reference

| Endpoint | Description | Time-Series | Fields |
|----------|-------------|-------------|--------|
| `/api/drift/history` | Historical drift metrics | [OK] | `psi_score`, `ks_p_value`, `js_divergence` |
| `/api/drift/metrics` | Current metrics (single point) | [Limited] | Current values only |
| `/api/drift/alerts` | Alert list | [OK] | `created_at` for time, `severity` for grouping |

## Time Field Format

The API returns timestamps in ISO 8601 format:
```
2026-01-18T10:00:00
```

Grafana's JSON API datasource automatically parses ISO format timestamps. Ensure your **Time Field** is set to `timestamp`.

## Additional Resources

- [Grafana JSON API Datasource Documentation](https://grafana.com/docs/grafana/latest/datasources/json-api/)
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Streamlit Dashboard](http://localhost:8501) - Alternative UI for monitoring
