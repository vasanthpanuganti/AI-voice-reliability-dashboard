# Deployment Guide

This guide covers deploying the AI Pipeline Resilience Dashboard to generic cloud platforms (for example a VM, container platform, or PaaS such as Render, Fly.io, or Vercel).

## Overview

The project consists of:
- **FastAPI Backend**: API endpoints for drift detection and rollback
- **Streamlit Dashboard**: Advanced Python dashboard for rich monitoring

## Container / PaaS Deployment (generic)

Most platforms that support Docker or generic Python apps follow the same pattern:

1. **Create a new service/app**
2. **Connect your GitHub repository** (or push a container image)
3. **Add a managed database** (PostgreSQL recommended for production) or mount a persistent volume if you want to keep using SQLite
4. **Configure Environment Variables** (see below)
5. **Expose Ports**:
   - API service: port `8000`
   - Dashboard service: port `8501`
6. **Deploy**: the platform will build the app from `requirements.txt` and run your chosen start command.

Example process manager commands:

- API: `python run_api.py`
- Dashboard: `python run_dashboard.py`

## Vercel Deployment (API only)

Vercel supports serverless functions, which can be used for the FastAPI backend with a web dashboard if you adapt the entrypoint.

### Steps:

1. **Install Vercel CLI** (optional, can use web interface):
   ```bash
   npm i -g vercel
   ```

2. **Deploy**:
   ```bash
   vercel
   ```
   Or connect your GitHub repository through the Vercel dashboard.

3. **Configure Environment Variables** in Vercel dashboard:
   - `DATABASE_URL`: PostgreSQL connection string (required - SQLite won't work on Vercel)
   - Other environment variables as needed

4. **Important**: 
   - The web dashboard (`frontend/index.html`) is automatically served at the root URL
   - The API entrypoint is configured in `api/index.py` and `vercel.json`
   - Vercel will automatically detect Python and install dependencies from `requirements.txt`

### Database Setup for Vercel

Since Vercel is serverless, you **must** use PostgreSQL or another managed database:
- Use Vercel Postgres (recommended)
- Or external services like Supabase or any other hosted Postgres
- Set `DATABASE_URL` in Vercel environment variables

## Environment Variables

### Required:
- `DATABASE_URL`: Database connection string
  - PostgreSQL: `postgresql://user:password@host:port/database`
  - SQLite (local only): `sqlite:///ai_resilience.db`

### Optional:
- `API_URL`: API base URL (for Streamlit dashboard, auto-detected for web dashboard)
- `PSI_WARNING_THRESHOLD`: Default 0.15
- `PSI_CRITICAL_THRESHOLD`: Default 0.25
- `PSI_EMERGENCY_THRESHOLD`: Default 0.40
- `DRIFT_WINDOW_MINUTES`: Default 15

## Testing Deployment

After deployment:

1. **Check API Health**:
   ```
   GET /health
   ```
   Should return: `{"status": "healthy"}`

2. **Access Dashboard**:
   - Web Dashboard: Visit root URL (`/`)
   - Streamlit Dashboard: Visit the Streamlit service URL

3. **Generate Sample Data** (if needed):
   ```bash
   python scripts/generate_sample_data.py
   ```

## Troubleshooting

### Dashboard shows "API: Offline"
- Check that the API service is running
- Verify `API_URL` environment variable (for Streamlit)
- Check CORS settings if accessing from different domain

### Database Connection Errors
- Verify `DATABASE_URL` is set correctly
- For PostgreSQL, ensure connection string uses `postgresql://` (not `postgres://`)
- Check database is accessible from deployment platform

### Metrics Not Showing
- Generate sample data first: `python scripts/generate_sample_data.py`
- Check API endpoints are responding: `/api/drift/metrics`
- Verify database has data in `query_logs` table
