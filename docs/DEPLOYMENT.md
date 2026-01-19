# Deployment Guide

This guide covers deploying the AI Pipeline Resilience Dashboard on Railway and Vercel.

## Overview

The project consists of:
- **FastAPI Backend**: API endpoints for drift detection and rollback
- **Web Dashboard**: HTML/JS dashboard served by FastAPI (works on Railway & Vercel)
- **Streamlit Dashboard**: Advanced Python dashboard (Railway only)

## Railway Deployment

### Option 1: Single Service (Web Dashboard)

The FastAPI backend automatically serves a web-based dashboard at the root URL (`/`).

1. **Create a new Railway project**
2. **Connect your GitHub repository**
3. **Add PostgreSQL Database** (if not using SQLite):
   - Click "New" → "Database" → "Add PostgreSQL"
   - Copy the `DATABASE_URL` from the database service
4. **Configure Environment Variables**:
   - `DATABASE_URL`: PostgreSQL connection string (if using PostgreSQL)
   - `API_URL`: Leave empty (dashboard auto-detects)
5. **Deploy**: Railway will automatically detect and deploy using `railway.json`

The dashboard will be available at your Railway URL (e.g., `https://your-app.up.railway.app`)

### Option 2: Two Services (API + Streamlit Dashboard)

For the full Streamlit dashboard experience:

#### Service 1: FastAPI Backend

1. Create a new service in Railway
2. Use the existing `railway.json` configuration
3. Set environment variables:
   - `DATABASE_URL`: PostgreSQL connection string
   - `PORT`: 8000 (default)

#### Service 2: Streamlit Dashboard

1. Create a second service in Railway
2. Use `railway-dashboard.json` configuration
3. Set environment variables:
   - `API_URL`: URL of your FastAPI service (e.g., `https://your-api.up.railway.app`)
   - `PORT`: 8501 (default)

**Note**: Streamlit requires a public URL. Railway will provide one automatically.

## Vercel Deployment

Vercel supports serverless functions, making it perfect for the FastAPI backend with the web dashboard.

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

Since Vercel is serverless, you **must** use PostgreSQL:
- Use Vercel Postgres (recommended)
- Or external services like Supabase, Railway Postgres, etc.
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
