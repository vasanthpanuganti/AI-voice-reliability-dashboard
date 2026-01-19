"""Streamlit Dashboard for AI Pipeline Resilience - Grafana-Style UI"""
import os
import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Configuration - Use environment variable for API URL in production
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Pipeline Resilience Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_grafana_theme():
    """Apply minimalistic sharp design with clean grays"""
    st.markdown("""
    <style>
    /* Minimalistic dark theme - flat colors */
    .stApp {
        background: #1a1a1a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling - flat color */
    [data-testid="stSidebar"] {
        background: #2a2a2a;
        border-right: 1px solid #404040;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    /* Headers - sharp typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Text colors */
    p, div, span, label {
        color: #e0e0e0;
    }
    
    /* Metric card - minimalistic sharp design (inline styles override) */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 0px;
        padding: 32px 24px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: box-shadow 0.2s;
        position: relative;
        display: block;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Alert cards - sharp design */
    .alert-panel {
        background: #1a1a1a;
        border-left: 4px solid;
        border-radius: 0px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .alert-warning { border-left-color: #F79420; }
    .alert-critical { border-left-color: #E24D42; }
    .alert-emergency { border-left-color: #8B0000; }
    
    /* Status indicator - sharp square */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 0px;
        margin-right: 8px;
    }
    
    .status-healthy { background-color: #73BF69; }
    .status-warning { background-color: #F79420; }
    .status-critical { background-color: #E24D42; }
    .status-emergency { background-color: #8B0000; }
    
    /* Button styling - flat colors, minimal radius */
    .stButton > button {
        background: #2a2a2a;
        color: white;
        border-radius: 2px;
        border: 1px solid #404040;
        padding: 10px 20px;
        font-weight: 600;
        transition: background 0.2s;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background: #3a3a3a;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Info boxes - flat colors */
    [data-testid="stInfo"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
    }
    
    [data-testid="stSuccess"] {
        background-color: #1a1a1a;
        border: 1px solid #73BF69;
    }
    
    [data-testid="stWarning"] {
        background-color: #1a1a1a;
        border: 1px solid #F79420;
    }
    
    [data-testid="stError"] {
        background-color: #1a1a1a;
        border: 1px solid #E24D42;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 6px;
    }
    
    /* Code blocks - sharp design */
    [data-testid="stCodeBlock"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 0px;
    }
    
    /* Chart container */
    .js-plotly-plot {
        background-color: #1a1a1a !important;
    }
    
    /* Scrollbar styling - minimal */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2a2a;
        border-radius: 0px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #404040;
    }
    
    /* Expander styling - sharp */
    [data-testid="stExpander"] {
        border: 1px solid #2a2a2a;
        border-radius: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", json_data: dict = None, timeout: int = 10):
    """Make API request with error handling and timeout"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error(f"API request timed out. Is the API server running at {API_BASE_URL}?")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Unable to connect to API at {API_BASE_URL}. Make sure the API server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API HTTP Error {e.response.status_code}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

@st.cache_data(ttl=30)
def get_drift_metrics():
    """Get current drift metrics"""
    return make_api_request("/api/drift/metrics")

@st.cache_data(ttl=30)
def get_alerts(status: str = "active"):
    """Get drift alerts"""
    return make_api_request(f"/api/drift/alerts?status={status}")

@st.cache_data(ttl=300)
def get_alert_diagnostics(alert_id: int):
    """Get detailed diagnostics for an alert"""
    return make_api_request(f"/api/drift/alerts/{alert_id}/diagnostics")

@st.cache_data(ttl=30)
def get_drift_history(limit: int = 100):
    """Get drift history"""
    return make_api_request(f"/api/drift/history?limit={limit}")

@st.cache_data(ttl=60)
def get_config_versions():
    """Get configuration versions"""
    return make_api_request("/api/rollback/versions")

@st.cache_data(ttl=30)
def get_rollback_history(limit: int = 50):
    """Get rollback history"""
    return make_api_request(f"/api/rollback/history?limit={limit}")

@st.cache_data(ttl=30)
def get_current_config():
    """Get current configuration"""
    return make_api_request("/api/rollback/current-config")

def execute_rollback(version_id: int, reason: str):
    """Execute rollback"""
    return make_api_request(
        "/api/rollback/execute",
        method="POST",
        json_data={"version_id": version_id, "reason": reason}
    )

def dismiss_alert(alert_id: int):
    """Dismiss an alert"""
    return make_api_request(f"/api/drift/alerts/{alert_id}/dismiss", method="POST")

def get_severity_color(severity: str):
    """Get Grafana-style color for severity level"""
    colors = {
        "normal": "#73BF69",   # Green
        "warning": "#F79420",  # Orange
        "critical": "#E24D42", # Red
        "emergency": "#8B0000" # Dark Red
    }
    return colors.get(severity.lower(), "#808080")

def create_metric_card(label: str, value: str, severity: str = "normal", subtitle: str = "", 
                       icon: str = "", progress: float = 0.0, max_value: float = 1.0, 
                       tooltip: str = ""):
    """Create metric card with inline styles for reliable rendering"""
    color = get_severity_color(severity)
    # Use rgba for semi-transparent backgrounds
    color_hex = color.lstrip('#')
    r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    badge_bg = f"rgba({r}, {g}, {b}, 0.15)"
    
    # Calculate progress percentage (clamped between 0 and 100)
    progress_pct = min(100, max(0, (progress / max_value * 100) if max_value > 0 else 0))
    
    # Severity status in plain language
    status_map = {
        "normal": "All Good",
        "warning": "Needs Attention",
        "critical": "Action Required",
        "emergency": "Urgent Action"
    }
    status_text = status_map.get(severity.lower(), severity.upper())
    
    tooltip_attr = f' title="{tooltip}"' if tooltip else ''
    
    # Simple structure with inline styles - no CSS variables
    card_html = f"""
    <div class="metric-card" style="background: #1a1a1a; border: 1px solid #2a2a2a; border-left: 4px solid {color}; padding: 32px 24px; margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.2); display: block;"{tooltip_attr}>
        <div style="font-size: 14px; color: #e0e0e0; font-weight: 500; margin-bottom: 8px; display: block;">{label}</div>
        <div style="font-size: 28px; font-weight: 700; color: {color}; margin: 12px 0; line-height: 1.2; display: block;">{value}</div>
        <div style="margin-top: 12px; height: 4px; background: #1a1a1a; border-radius: 0; overflow: hidden; display: block;">
            <div style="height: 100%; width: {progress_pct}%; background: {color}; transition: width 0.3s; display: block;"></div>
        </div>
        <span style="display: inline-block; padding: 4px 12px; border-radius: 2px; font-size: 10px; font-weight: 600; text-transform: uppercase; margin-top: 12px; background: {badge_bg}; color: {color};">
            {status_text}
        </span>
        {f'<div style="color: #a0a0a0; font-size: 11px; margin-top: 8px; display: block;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    return card_html

def create_dark_chart_layout(title: str, yaxis_title: str, height: int = 400):
    """Create Grafana-style dark chart layout"""
    return {
        "title": {
            "text": title,
            "font": {"color": "#FFFFFF", "size": 18, "family": "Inter, sans-serif"},
            "x": 0.05,
            "xanchor": "left"
        },
        "xaxis": {
            "gridcolor": "#404040",
            "gridwidth": 1,
            "color": "#B7B7B7",
            "title": {"font": {"color": "#B7B7B7", "size": 12}},
            "showgrid": True,
            "zeroline": False
        },
        "yaxis": {
            "gridcolor": "#404040",
            "gridwidth": 1,
            "color": "#B7B7B7",
            "title": {"text": yaxis_title, "font": {"color": "#B7B7B7", "size": 12}},
            "showgrid": True,
            "zeroline": False
        },
        "plot_bgcolor": "#1e1e1e",
        "paper_bgcolor": "#1e1e1e",
        "font": {"color": "#B7B7B7", "family": "Inter, sans-serif"},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#2d2d2d",
            "bordercolor": "#404040",
            "font": {"color": "#FFFFFF"}
        },
        "legend": {
            "font": {"color": "#B7B7B7"},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "#404040",
            "borderwidth": 1
        },
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50}
    }

def main():
    """Main dashboard"""
    # Apply Grafana theme
    apply_grafana_theme()
    
    # Header with status indicator
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <h1 style="margin: 0; color: #FFFFFF;">AI Pipeline Resilience Dashboard</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p style="color: #B7B7B7; margin-top: -10px;">Drift Detection and Rollback System for Healthcare AI Pipelines</p>', unsafe_allow_html=True)
    
    with col_header2:
        # API status indicator
        try:
            health = make_api_request("/health", timeout=2)
            status_color = "#73BF69" if health else "#E24D42"
            status_text = "Healthy" if health else "Offline"
        except:
            status_color = "#E24D42"
            status_text = "Offline"
        
        st.markdown(f"""
        <div style="text-align: right; margin-top: 20px;">
            <span class="status-indicator status-healthy" style="background: {status_color};"></span>
            <span style="color: #B7B7B7; font-size: 14px;">API: {status_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="padding: 20px 0;">
        <h2 style="color: #FFFFFF; margin: 0;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Page",
        ["Drift Detection", "Rollback Control", "System Overview"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 60, 30)
    
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.markdown(f"**API:** `{API_BASE_URL}`")
    
    if page == "Drift Detection":
        show_drift_detection_page()
    elif page == "Rollback Control":
        show_rollback_page()
    elif page == "System Overview":
        show_system_overview()

def show_drift_detection_page():
    """Drift Detection Dashboard with Grafana-style UI"""
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="color: #FFFFFF; margin-bottom: 5px;">Drift Detection</h2>
        <p style="color: #B7B7B7; margin: 0;">Monitor input, output, and embedding space shifts in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current metrics with loading state
    with st.spinner("Loading drift metrics..."):
        metrics = get_drift_metrics()
    
    if not metrics:
        st.error("Unable to fetch drift metrics. Make sure the API is running at http://localhost:8000")
        st.info("**Troubleshooting:**\n- Check if API is running: `python run_api.py`\n- Verify API health: http://localhost:8000/health\n- Check API logs for errors")
        return
    
    # Enhanced Metric Cards
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        psi_score = metrics.get("psi_score", 0)
        severity = get_psi_severity(psi_score)
        st.markdown(create_metric_card(
            label="Input Shift Detection",
            value=f"{psi_score:.4f}" if psi_score > 0 else "0.00",
            severity=severity,
            subtitle="How much patient questions have changed",
            icon="",
            progress=min(psi_score, 0.4),  # Scale for display (0-0.4 range)
            max_value=0.4,
            tooltip="Measures how different current patient questions are compared to the baseline. Lower is better."
        ), unsafe_allow_html=True)
    
    with col2:
        ks_p_value = metrics.get("ks_p_value", 1.0)
        severity = get_ks_severity(ks_p_value)
        # For KS, lower p-value means more drift, so invert for progress bar
        progress_value = 1.0 - min(ks_p_value, 1.0)
        st.markdown(create_metric_card(
            label="Output Quality Check",
            value=f"{ks_p_value:.4f}",
            severity=severity,
            subtitle="How reliable AI responses are",
            icon="",
            progress=progress_value,
            max_value=1.0,
            tooltip="Measures how consistent AI responses are. Values closer to 1.0 mean responses are consistent and reliable."
        ), unsafe_allow_html=True)
    
    with col3:
        js_divergence = metrics.get("js_divergence", 0)
        severity = get_js_severity(js_divergence)
        st.markdown(create_metric_card(
            label="Understanding Accuracy",
            value=f"{js_divergence:.4f}" if js_divergence > 0 else "0.00",
            severity=severity,
            subtitle="How well AI understands queries",
            icon="",
            progress=min(js_divergence, 0.3),  # Scale for display (0-0.3 range)
            max_value=0.3,
            tooltip="Measures how well the AI system understands patient questions compared to baseline. Lower values mean better understanding."
        ), unsafe_allow_html=True)
    
    with col4:
        sample_size = metrics.get("sample_size", 0)
        st.markdown(create_metric_card(
            label="Active Queries",
            value=f"{sample_size:,}" if sample_size > 0 else "0",
            severity="normal",
            subtitle="Patient queries analyzed recently",
            icon="",
            progress=min(sample_size / 100, 1.0) if sample_size > 0 else 0,  # Scale to 100 queries = 100%
            max_value=1.0,
            tooltip="Number of patient queries processed in the current monitoring window."
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Alerts Section
    st.markdown("### Active Alerts")
    alerts = get_alerts(status="active")
    
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        
        for _, alert in alerts_df.iterrows():
            severity = alert.get("severity", "warning")
            color = get_severity_color(severity)
            alert_id = alert["id"]
            alert_class = f"alert-{severity.lower()}"
            
            diagnostics = get_alert_diagnostics(alert_id)
            
            with st.expander(
                f"ALERT: {alert['metric_name'].upper()} - {severity.upper()} (Value: {alert['metric_value']:.4f})",
                expanded=True
            ):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Metric:** `{alert['metric_name']}`")
                    st.markdown(f"**Current Value:** `{alert['metric_value']:.4f}`")
                    st.markdown(f"**Threshold:** `{alert['threshold_value']:.4f}`")
                    st.markdown(f"**Type:** {alert['metric_type']}")
                
                with col2:
                    created_at = datetime.fromisoformat(alert["created_at"].replace("Z", "+00:00"))
                    st.markdown(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f'<span style="color:{color};font-weight:bold;font-size:18px;">{severity.upper()}</span>', unsafe_allow_html=True)
                
                with col3:
                    if st.button("Dismiss", key=f"dismiss_{alert_id}"):
                        with st.spinner("Dismissing alert..."):
                            result = dismiss_alert(alert_id)
                            if result:
                                st.success("Alert dismissed")
                                st.cache_data.clear()
                                time.sleep(0.5)
                                st.rerun()
                
                if diagnostics and "diagnostics" in diagnostics:
                    diag = diagnostics["diagnostics"]
                    st.markdown("---")
                    st.markdown("#### Why This Alert Occurred")
                    st.info(f"**What it means:** {diag.get('what_it_means', 'No explanation available')}")
                    
                    if "baseline_period" in diag and "current_period" in diag:
                        col_period1, col_period2 = st.columns(2)
                        with col_period1:
                            st.markdown(f"**Baseline Period:** {diag['baseline_period']}")
                        with col_period2:
                            st.markdown(f"**Current Period:** {diag['current_period']}")
                    
                    if "category_shifts" in diag and diag["category_shifts"]:
                        st.markdown("#### Category Distribution Shifts")
                        shifts_df = pd.DataFrame(diag["category_shifts"])
                        st.dataframe(shifts_df, use_container_width=True)
                    
                    if "recommendations" in diag and diag["recommendations"]:
                        st.markdown("#### Recommended Actions")
                        for i, rec in enumerate(diag["recommendations"], 1):
                            st.markdown(f"{i}. {rec}")
                
                st.divider()
    else:
        st.success("No active alerts - System operating normally")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Historical Trends with Enhanced Charts
    st.markdown("### Drift Trends")
    st.caption("Historical drift metrics over time with threshold indicators")
    history = get_drift_history(limit=100)
    
    if not history:
        st.info("No historical data available yet. Drift metrics will appear here as data is collected.")
        return
    
    history_df = pd.DataFrame(history)
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    
    # PSI Score Chart
    fig = go.Figure()
    
    if "psi_score" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=history_df["timestamp"],
            y=history_df["psi_score"],
            name="PSI Score",
            mode="lines+markers",
            line=dict(color="#3274D9", width=2),
            marker=dict(size=4, color="#3274D9"),
            fill="tonexty",
            fillcolor="rgba(50, 116, 217, 0.1)"
        ))
    
    # Threshold lines
    fig.add_hline(
        y=0.15, 
        line_dash="dash", 
        line_color="#F79420", 
        annotation_text="Warning (0.15)",
        annotation_position="right"
    )
    fig.add_hline(
        y=0.25, 
        line_dash="dash", 
        line_color="#E24D42", 
        annotation_text="Critical (0.25)",
        annotation_position="right"
    )
    fig.add_hline(
        y=0.40, 
        line_dash="dash", 
        line_color="#8B0000", 
        annotation_text="Emergency (0.40)",
        annotation_position="right"
    )
    
    fig.update_layout(**create_dark_chart_layout("PSI Score Over Time", "PSI Score", 450))
    st.plotly_chart(fig, use_container_width=True)
    
    # JS Divergence Chart
    fig2 = go.Figure()
    if "js_divergence" in history_df.columns:
        fig2.add_trace(go.Scatter(
            x=history_df["timestamp"],
            y=history_df["js_divergence"],
            name="JS Divergence",
            mode="lines+markers",
            line=dict(color="#B877D9", width=2),
            marker=dict(size=4, color="#B877D9"),
            fill="tonexty",
            fillcolor="rgba(184, 119, 217, 0.1)"
        ))
    
    fig2.add_hline(y=0.1, line_dash="dash", line_color="#F79420", annotation_text="Warning (0.1)", annotation_position="right")
    fig2.add_hline(y=0.2, line_dash="dash", line_color="#E24D42", annotation_text="Critical (0.2)", annotation_position="right")
    fig2.add_hline(y=0.3, line_dash="dash", line_color="#8B0000", annotation_text="Emergency (0.3)", annotation_position="right")
    
    fig2.update_layout(**create_dark_chart_layout("Jensen-Shannon Divergence Over Time", "JS Divergence", 450))
    st.plotly_chart(fig2, use_container_width=True)

def show_rollback_page():
    """Rollback Control Dashboard"""
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="color: #FFFFFF; margin-bottom: 5px;">Rollback Control</h2>
        <p style="color: #B7B7B7; margin: 0;">Manage configuration versions and execute rollbacks when drift is detected</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current Configuration
    st.markdown("### Current Configuration")
    current_config = get_current_config()
    
    if current_config:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div style="background: #2d2d2d; border: 1px solid #404040; border-radius: 8px; padding: 20px;">
                <p style="color: #B7B7B7; margin: 8px 0;"><strong style="color: #FFFFFF;">Embedding Model:</strong> <code style="background: #1e1e1e; padding: 2px 8px; border-radius: 4px;">{current_config.get('embedding_model', 'N/A')}</code></p>
                <p style="color: #B7B7B7; margin: 8px 0;"><strong style="color: #FFFFFF;">Similarity Threshold:</strong> <code style="background: #1e1e1e; padding: 2px 8px; border-radius: 4px;">{current_config.get('similarity_threshold', 'N/A')}</code></p>
                <p style="color: #B7B7B7; margin: 8px 0;"><strong style="color: #FFFFFF;">Confidence Threshold:</strong> <code style="background: #1e1e1e; padding: 2px 8px; border-radius: 4px;">{current_config.get('confidence_threshold', 'N/A')}</code></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.info("**Active Configuration**\n\nThis configuration is currently used by the AI pipeline.")
    else:
        st.warning("No configuration found")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Configuration Versions
    st.markdown("### Configuration Versions")
    versions = get_config_versions()
    
    if versions:
        versions_df = pd.DataFrame(versions)
        
        col1, col2 = st.columns(2)
        with col1:
            show_known_good = st.checkbox("Show only known-good versions", value=False)
        with col2:
            if st.button("Create Snapshot"):
                result = make_api_request("/api/rollback/snapshot", method="POST", json_data={})
                if result:
                    st.success("Snapshot created!")
                    time.sleep(0.5)
                    st.rerun()
        
        if show_known_good:
            versions_df = versions_df[versions_df["is_known_good"] == True]
        
        for _, version in versions_df.head(20).iterrows():
            with st.expander(f"Version {version['id']}: {version.get('version_label', 'N/A')} - {version['snapshot_timestamp'][:19]}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Embedding Model:** {version['embedding_model']}")
                    st.write(f"**Similarity Threshold:** {version['similarity_threshold']}")
                    st.write(f"**Confidence Threshold:** {version['confidence_threshold']}")
                    if version.get("is_known_good"):
                        st.success("Known Good Version")
                
                with col2:
                    with st.form(key=f"rollback_form_{version['id']}"):
                        reason = st.text_input("Rollback reason:", value="Manual rollback via dashboard", key=f"reason_{version['id']}")
                        if st.form_submit_button("Rollback", use_container_width=True):
                            if reason:
                                with st.spinner("Executing rollback..."):
                                    result = execute_rollback(version["id"], reason)
                                    if result:
                                        st.success(f"Rollback executed! Status: {result.get('status')}")
                                        st.cache_data.clear()
                                        time.sleep(1)
                                        st.rerun()
                            else:
                                st.warning("Please provide a rollback reason")
    else:
        st.info("No configuration versions found")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Rollback History
    st.markdown("### Rollback History")
    rollback_history = get_rollback_history(limit=20)
    
    if rollback_history:
        history_df = pd.DataFrame(rollback_history)
        
        status_colors = {
            "success": "#73BF69",
            "failed": "#E24D42",
            "pending": "#F79420",
            "in_progress": "#3274D9"
        }
        
        for _, event in history_df.iterrows():
            status = event.get("status", "pending")
            color = status_colors.get(status.lower(), "#808080")
            
            st.markdown(f"""
            <div style="background: #2d2d2d; border: 1px solid #404040; border-radius: 6px; padding: 15px; margin: 10px 0;">
                <p style="color: #FFFFFF; margin: 5px 0; font-weight: 600;">Rollback #{event['id']} - {event['executed_at'][:19]}</p>
                <p style="color: #B7B7B7; margin: 5px 0;">Type: {event['trigger_type']} | Status: <span style="color: {color};">{status}</span></p>
                <p style="color: #B7B7B7; margin: 5px 0;">Reason: {event.get('trigger_reason', 'N/A')}</p>
                <p style="color: #B7B7B7; margin: 5px 0;">Restored Version: {event['restored_version_id']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No rollback events")

def show_system_overview():
    """System Overview Page"""
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="color: #FFFFFF; margin-bottom: 5px;">System Overview</h2>
        <p style="color: #B7B7B7; margin: 0;">Comprehensive view of system status, alerts, and dataset information</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    alerts = get_alerts(status="active")
    active_alerts_count = len(alerts) if alerts else 0
    alert_color = "#E24D42" if active_alerts_count > 0 else "#73BF69"
    
    with col1:
        st.markdown(create_metric_card(
            "Active Alerts",
            str(active_alerts_count),
            "critical" if active_alerts_count > 0 else "normal",
            "Current alert count"
        ), unsafe_allow_html=True)
    
    versions = get_config_versions()
    version_count = len(versions) if versions else 0
    
    with col2:
        st.markdown(create_metric_card(
            "Config Versions",
            str(version_count),
            "normal",
            "Total snapshots"
        ), unsafe_allow_html=True)
    
    rollback_history = get_rollback_history(limit=1000)
    rollback_count = len(rollback_history) if rollback_history else 0
    
    with col3:
        st.markdown(create_metric_card(
            "Total Rollbacks",
            str(rollback_count),
            "normal",
            "Rollback events"
        ), unsafe_allow_html=True)
    
    with col4:
        metrics = get_drift_metrics()
        sample_size = metrics.get("sample_size", 0) if metrics else 0
        st.markdown(create_metric_card(
            "Queries Processed",
            f"{sample_size:,}",
            "normal",
            "Current window"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Alert Distribution
    if alerts:
        st.markdown("### Alert Distribution by Severity")
        alerts_df = pd.DataFrame(alerts)
        severity_counts = alerts_df["severity"].value_counts()
        
        fig = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            color_discrete_map={
                "warning": "#F79420",
                "critical": "#E24D42",
                "emergency": "#8B0000"
            }
        )
        fig.update_layout(**create_dark_chart_layout("Alert Distribution", "Count", 400))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

def get_psi_severity(psi_score: float) -> str:
    """Get severity for PSI score"""
    if psi_score >= 0.40:
        return "emergency"
    elif psi_score >= 0.25:
        return "critical"
    elif psi_score >= 0.15:
        return "warning"
    return "normal"

def get_ks_severity(p_value: float) -> str:
    """Get severity for KS p-value (lower is worse)"""
    if p_value <= 0.001:
        return "emergency"
    elif p_value <= 0.01:
        return "critical"
    elif p_value <= 0.05:
        return "warning"
    return "normal"

def get_js_severity(js_div: float) -> str:
    """Get severity for JS divergence"""
    if js_div >= 0.3:
        return "emergency"
    elif js_div >= 0.2:
        return "critical"
    elif js_div >= 0.1:
        return "warning"
    return "normal"

if __name__ == "__main__":
    main()
