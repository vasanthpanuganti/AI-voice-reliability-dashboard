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


def safe_parse_datetime(dt_string):
    """Safely parse datetime string from API with fallback."""
    if not dt_string:
        return datetime.now()
    try:
        return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        try:
            return datetime.strptime(dt_string[:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
            return datetime.now()

st.set_page_config(
    page_title="AI Pipeline Resilience Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def _get_grafana_theme_css():
    """Return cached CSS string to avoid recomputation."""
    return """<style>
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
    
    /* Code blocks - sharp design with dark theme for readability */
    [data-testid="stCodeBlock"] {
        background-color: #2a2a2a !important;
        border: 1px solid #404040 !important;
        border-radius: 0px;
    }
    
    [data-testid="stCodeBlock"] pre {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    
    [data-testid="stCodeBlock"] code {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    
    /* Ensure all code elements have readable text */
    code {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        padding: 2px 6px;
        border-radius: 2px;
    }
    
    pre {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    
    pre code {
        background-color: transparent !important;
        color: #e0e0e0 !important;
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
    
    /* Info tooltip styles */
    .info-tooltip-container:hover .info-tooltip-content {
        visibility: visible !important;
    }
    
    .info-btn {
        transition: background 0.2s;
    }
    
    .info-btn:hover {
        background: #5a9cf0 !important;
    }
    
    .info-tooltip-content::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: #2d2d2d transparent transparent transparent;
    }
    </style>"""


def apply_grafana_theme():
    """Apply minimalistic sharp design with clean grays."""
    st.markdown(_get_grafana_theme_css(), unsafe_allow_html=True)


# ============================================================================
# INFO TOOLTIP DEFINITIONS
# ============================================================================
INFO_TOOLTIPS = {
    # Drift Metrics
    "psi_score": {
        "title": "Population Stability Index (PSI)",
        "description": "Measures how much the distribution of patient queries has changed compared to the baseline period.",
        "interpretation": "• < 0.10: No significant change\n• 0.10-0.25: Moderate change (Warning)\n• > 0.25: Significant change (Critical)",
        "action": "High PSI indicates patients are asking different types of questions than expected. Review query categories for shifts."
    },
    "ks_p_value": {
        "title": "Kolmogorov-Smirnov Test (p-value)",
        "description": "Statistical test measuring if AI confidence scores have changed significantly from baseline.",
        "interpretation": "• > 0.05: Normal (no significant change)\n• 0.01-0.05: Warning\n• < 0.01: Critical (significant change)",
        "action": "Low p-value suggests AI confidence is behaving differently. May indicate model degradation or data quality issues."
    },
    "js_divergence": {
        "title": "Jensen-Shannon Divergence",
        "description": "Measures how much the AI's internal understanding (embeddings) has shifted from baseline.",
        "interpretation": "• < 0.10: Normal\n• 0.10-0.20: Warning\n• > 0.20: Critical",
        "action": "High JS divergence indicates the AI is 'thinking' about queries differently. May require model retraining or embedding updates."
    },
    "sample_size": {
        "title": "Sample Size",
        "description": "Number of patient queries analyzed in the current monitoring window (typically 15 minutes).",
        "interpretation": "More samples = more reliable drift metrics. Minimum 100 samples recommended for accurate detection.",
        "action": "If sample size is low, metrics may be unreliable. Wait for more data to accumulate."
    },
    # Confidence Routing
    "confidence_threshold": {
        "title": "Confidence Thresholds",
        "description": "Determines how AI responses are routed based on model confidence scores.",
        "interpretation": "• High (≥0.85): AI responds directly\n• Medium (0.70-0.85): May need validation\n• Low (0.50-0.70): Likely needs human\n• Reject (<0.30): Always escalate",
        "action": "Adjust thresholds based on acceptable error rates. Lower thresholds = more AI responses, higher risk."
    },
    "risk_penalty": {
        "title": "Risk Level Penalties",
        "description": "Confidence score is reduced for sensitive topics to ensure safer routing.",
        "interpretation": "• Low Risk: No penalty\n• Medium Risk: -10% confidence\n• High Risk: -20% confidence\n• Critical Risk: -35% confidence",
        "action": "Penalties ensure high-risk queries (medications, symptoms) require higher confidence to pass."
    },
    "topic_classification": {
        "title": "Topic Classification",
        "description": "Automatically detects sensitive topics in patient queries using pattern matching.",
        "interpretation": "Topics include: Medication, Clinical Symptoms, Billing Disputes, Personal Health Info",
        "action": "Critical topics (symptoms) always route to humans regardless of confidence."
    },
    # Rollback
    "known_good_version": {
        "title": "Known-Good Version",
        "description": "A configuration version that has been verified to work well based on performance metrics.",
        "interpretation": "Known-good versions are preferred targets for automated rollbacks.",
        "action": "Mark stable versions as 'known-good' after confirming good performance metrics."
    },
    "rollback_trigger": {
        "title": "Automated Rollback Triggers",
        "description": "System automatically rolls back when critical conditions are met.",
        "interpretation": "• Emergency Alert: Immediate rollback\n• Sustained Critical: 3+ critical alerts in 15 min\n• Confidence Collapse: 25% drop from baseline",
        "action": "Cooldown period (30 min) prevents repeated rollbacks. System restores best known-good version."
    },
    # Segment Analysis
    "segment_drift": {
        "title": "Segment-Level Drift",
        "description": "Monitors drift for specific patient populations or query categories separately.",
        "interpretation": "Catches issues affecting specific groups even when aggregate metrics look fine.",
        "action": "If one segment shows drift while others are normal, investigate that specific category."
    },
    "distribution_shift": {
        "title": "Distribution Shift",
        "description": "How much this category's share of queries has changed from baseline.",
        "interpretation": "• < 5%: Normal variation\n• 5-10%: Notable shift\n• > 10%: Significant shift",
        "action": "Large shifts indicate changing patient behavior or possible data collection issues."
    },
}


def create_info_button(key: str, inline: bool = True) -> str:
    """Create an info button with tooltip explanation."""
    info = INFO_TOOLTIPS.get(key, {})
    if not info:
        return ""
    
    title = info.get("title", key)
    description = info.get("description", "")
    interpretation = info.get("interpretation", "").replace("\n", "<br>")
    action = info.get("action", "")
    
    tooltip_content = f"""
        <strong style='color: #3274D9;'>{title}</strong><br><br>
        <strong>What it measures:</strong><br>{description}<br><br>
        <strong>How to read it:</strong><br>{interpretation}<br><br>
        <strong>What to do:</strong><br>{action}
    """
    
    if inline:
        return f"""
        <span class="info-tooltip-container" style="position: relative; display: inline-block; margin-left: 6px;">
            <span class="info-btn" style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #3274D9;
                color: white;
                font-size: 10px;
                font-weight: bold;
                cursor: help;
                font-style: normal;
            ">i</span>
            <span class="info-tooltip-content" style="
                visibility: hidden;
                position: absolute;
                z-index: 1000;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                width: 300px;
                background: #2d2d2d;
                color: #e0e0e0;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #404040;
                font-size: 12px;
                line-height: 1.5;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                text-align: left;
            ">{tooltip_content}</span>
        </span>
        """
    else:
        return tooltip_content


def show_info_expander(key: str, label: str = "Info: What does this mean?"):
    """Show info in a Streamlit expander."""
    info = INFO_TOOLTIPS.get(key, {})
    if not info:
        return
    
    with st.expander(label, expanded=False):
        st.markdown(f"**{info.get('title', key)}**")
        st.markdown(info.get('description', ''))
        st.markdown("**How to read it:**")
        # Use markdown with dark background styling instead of st.code for better readability
        interpretation = info.get('interpretation', '')
        st.markdown(
            f'<div style="background-color: #2a2a2a; border: 1px solid #404040; padding: 15px; margin: 10px 0; color: #e0e0e0; font-family: monospace; white-space: pre-line; line-height: 1.6;">{interpretation}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**What to do:** {info.get('action', '')}")


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


def make_api_request_with_retry(endpoint: str, method: str = "GET", json_data: dict = None,
                                 timeout: int = 10, max_retries: int = 2):
    """Make API request with automatic retry on transient failures."""
    for attempt in range(max_retries):
        result = make_api_request(endpoint, method, json_data, timeout)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            time.sleep(1)
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

@st.cache_data(ttl=30)
def get_segment_drift(segment_by: str = "query_category"):
    """Get segment-level drift metrics"""
    return make_api_request(f"/api/drift/segments?segment_by={segment_by}")

@st.cache_data(ttl=30)
def get_drift_summary():
    """Get comprehensive drift summary"""
    return make_api_request("/api/drift/summary")

@st.cache_data(ttl=30)
def get_trigger_status():
    """Get automated rollback trigger status"""
    return make_api_request("/api/rollback/triggers/status")

@st.cache_data(ttl=30)
def get_routing_stats(hours: int = 24):
    """Get confidence routing statistics"""
    return make_api_request(f"/api/routing/stats?hours={hours}")

@st.cache_data(ttl=60)
def get_routing_thresholds():
    """Get confidence routing thresholds"""
    return make_api_request("/api/routing/thresholds")

@st.cache_data(ttl=60)
def get_sensitive_topics():
    """Get sensitive topic configuration"""
    return make_api_request("/api/routing/topics")

def evaluate_query_routing(query: str, ai_response: str, confidence: float):
    """Evaluate a query for routing decision"""
    return make_api_request(
        "/api/routing/evaluate",
        method="POST",
        json_data={
            "query": query,
            "ai_response": ai_response,
            "confidence_score": confidence
        }
    )

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
        ["Drift Detection", "Segment Analysis", "Confidence Routing", "Rollback Control", "System Overview"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Refresh Controls")

    col_refresh1, col_refresh2 = st.sidebar.columns(2)
    with col_refresh1:
        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_refresh2:
        auto_refresh = st.checkbox("Auto", value=False, help="Enable auto-refresh")

    if auto_refresh:
        refresh_interval = st.sidebar.slider("Interval (seconds)", 10, 60, 30)
        time.sleep(refresh_interval)
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.markdown(f"**API:** `{API_BASE_URL}`")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Help")
    with st.sidebar.expander("Metric Thresholds"):
        st.markdown("""
        **PSI (Input Drift)**
        - Normal: < 0.10
        - Warning: 0.10 - 0.25
        - Critical: > 0.25
        
        **KS p-value (Output Quality)**
        - Normal: > 0.05
        - Warning: 0.01 - 0.05  
        - Critical: < 0.01
        
        **JS Divergence (Understanding)**
        - Normal: < 0.10
        - Warning: 0.10 - 0.20
        - Critical: > 0.20
        """)
    
    with st.sidebar.expander("Routing Rules"):
        st.markdown("""
        **Confidence Thresholds**
        - ≥0.85: AI responds directly
        - 0.70-0.85: May need validation
        - 0.50-0.70: Likely needs human
        - <0.30: Always escalate
        
        **Risk Penalties**
        - Critical topics: -35%
        - High risk: -20%
        - Medium risk: -10%
        """)
    
    if page == "Drift Detection":
        show_drift_detection_page()
    elif page == "Segment Analysis":
        show_segment_analysis_page()
    elif page == "Confidence Routing":
        show_confidence_routing_page()
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
    
    # Enhanced Metric Cards with info buttons
    st.markdown("### Key Metrics")
    show_info_expander("psi_score", "ℹ️ Understanding Drift Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        psi_score = metrics.get("psi_score", 0)
        severity = get_psi_severity(psi_score)
        st.markdown(create_metric_card(
            label=f"Input Shift (PSI)",
            value=f"{psi_score:.4f}" if psi_score > 0 else "0.00",
            severity=severity,
            subtitle="How much patient questions have changed",
            icon="",
            progress=min(psi_score, 0.4),
            max_value=0.4,
            tooltip="PSI measures query distribution shift. <0.10 normal, 0.10-0.25 warning, >0.25 critical"
        ), unsafe_allow_html=True)

    with col2:
        ks_p_value = metrics.get("ks_p_value", 1.0)
        severity = get_ks_severity(ks_p_value)
        progress_value = 1.0 - min(ks_p_value, 1.0)
        st.markdown(create_metric_card(
            label=f"Output Quality (KS)",
            value=f"{ks_p_value:.4f}",
            severity=severity,
            subtitle="How reliable AI responses are",
            icon="",
            progress=progress_value,
            max_value=1.0,
            tooltip="KS p-value tests confidence consistency. >0.05 normal, 0.01-0.05 warning, <0.01 critical"
        ), unsafe_allow_html=True)

    with col3:
        js_divergence = metrics.get("js_divergence", 0)
        severity = get_js_severity(js_divergence)
        st.markdown(create_metric_card(
            label=f"Understanding (JS)",
            value=f"{js_divergence:.4f}" if js_divergence > 0 else "0.00",
            severity=severity,
            subtitle="How well AI understands queries",
            icon="",
            progress=min(js_divergence, 0.3),
            max_value=0.3,
            tooltip="JS divergence measures embedding shift. <0.10 normal, 0.10-0.20 warning, >0.20 critical"
        ), unsafe_allow_html=True)

    with col4:
        sample_size = metrics.get("sample_size") or 0
        
        # Automatically refresh queries if active queries is zero
        if sample_size == 0:
            # Use session state to prevent multiple simultaneous refresh attempts
            if "query_refresh_attempted" not in st.session_state:
                st.session_state.query_refresh_attempted = True
                try:
                    # Automatically refresh queries
                    refresh_response = make_api_request(
                        "/api/drift/refresh-queries", method="POST", timeout=30
                    )
                    if refresh_response:
                        st.info(
                            f"Queries refreshed successfully: {refresh_response.get('message', 'Success')}"
                        )
                        # Clear cache and rerun to show updated metrics
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.warning(f"Warning: Could not automatically refresh queries: {str(e)}")
            else:
                # Reset after a delay (handled by rerun)
                if "query_refresh_delay" not in st.session_state:
                    st.session_state.query_refresh_delay = time.time()
                elif time.time() - st.session_state.query_refresh_delay > 5:
                    # Reset after 5 seconds
                    del st.session_state.query_refresh_attempted
                    del st.session_state.query_refresh_delay
        
        st.markdown(create_metric_card(
            label="Active Queries",
            value=f"{sample_size:,}" if sample_size > 0 else "0",
            severity="normal" if sample_size > 0 else "warning",
            subtitle="Patient queries analyzed recently",
            icon="",
            progress=min(sample_size / 100, 1.0) if sample_size > 0 else 0,
            max_value=1.0,
            tooltip="Number of queries in current 15-min window. More samples = more reliable metrics."
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
                    created_at = safe_parse_datetime(alert.get("created_at"))
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

def show_segment_analysis_page():
    """Segment-Level Drift Analysis"""
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="color: #FFFFFF; margin-bottom: 5px;">Segment Analysis</h2>
        <p style="color: #B7B7B7; margin: 0;">Monitor drift at segment level to catch issues affecting specific patient populations</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_info_expander("segment_drift", "Info: Why segment-level monitoring matters")
    
    # Segment selection
    segment_by = st.selectbox(
        "Segment By",
        ["query_category", "department", "patient_population"],
        format_func=lambda x: x.replace("_", " ").title(),
        help="Choose how to group queries for segment-level drift analysis"
    )
    
    with st.spinner("Analyzing segments..."):
        segment_data = get_segment_drift(segment_by)
    
    if not segment_data:
        st.warning("Unable to fetch segment drift data. Make sure the API is running.")
        return
    
    # Overall health
    overall_health = segment_data.get("overall_health", "unknown")
    health_colors = {"healthy": "#73BF69", "degraded": "#F79420", "critical": "#E24D42"}
    health_color = health_colors.get(overall_health, "#808080")
    
    st.markdown(f"""
    <div style="background: #2d2d2d; border-left: 4px solid {health_color}; padding: 20px; margin: 20px 0;">
        <h3 style="color: #FFFFFF; margin: 0;">Overall Segment Health: 
            <span style="color: {health_color};">{overall_health.upper()}</span>
        </h3>
        <p style="color: #B7B7B7; margin-top: 10px;">
            Total Segments: {segment_data.get('total_segments', 0)} | 
            Healthy: {segment_data.get('healthy_segments', 0)} | 
            Degraded: {segment_data.get('degraded_segments', 0)} | 
            Critical: {segment_data.get('critical_segments', 0)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Segment details
    st.markdown("### Segment Details")
    
    segments = segment_data.get("segments", {})
    
    if segments:
        for segment_name, segment_info in segments.items():
            status = segment_info.get("status", "unknown")
            status_color = health_colors.get(status, "#808080")
            
            with st.expander(f"{segment_name.upper()} - {status.upper()}", expanded=status != "healthy"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Baseline Count", segment_info.get("baseline_count", 0))
                    st.metric("Current Count", segment_info.get("window_count", 0))
                
                with col2:
                    baseline_pct = segment_info.get("baseline_percentage", 0)
                    current_pct = segment_info.get("window_percentage", 0)
                    st.metric("Baseline %", f"{baseline_pct:.1f}%")
                    st.metric("Current %", f"{current_pct:.1f}%")
                
                with col3:
                    shift = segment_info.get("distribution_shift", 0)
                    shift_color = "#73BF69" if abs(shift) < 5 else "#F79420" if abs(shift) < 10 else "#E24D42"
                    st.markdown(f"""
                    <div style="padding: 20px; text-align: center;">
                        <div style="font-size: 12px; color: #B7B7B7;">Distribution Shift</div>
                        <div style="font-size: 32px; font-weight: bold; color: {shift_color};">
                            {'+' if shift > 0 else ''}{shift:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Input drift details
                input_drift = segment_info.get("input_drift", {})
                if input_drift:
                    st.markdown("**Input Drift:**")
                    st.write(f"PSI Score: {input_drift.get('psi_score', 0):.4f}")
                    if input_drift.get("drift_detected"):
                        st.warning(f"Drift Detected! Severity: {input_drift.get('severity', 'unknown')}")
    else:
        st.info("No segment data available. Ensure you have query data with category information.")
    
    # Time window info
    window = segment_data.get("window", {})
    baseline = segment_data.get("baseline", {})
    if window and baseline:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Baseline Period:** {baseline.get('start', 'N/A')[:19]} to {baseline.get('end', 'N/A')[:19]}")
        with col2:
            st.markdown(f"**Current Window:** {window.get('start', 'N/A')[:19]} to {window.get('end', 'N/A')[:19]}")


def show_confidence_routing_page():
    """Confidence-Based Routing Dashboard"""
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="color: #FFFFFF; margin-bottom: 5px;">Confidence-Based Routing</h2>
        <p style="color: #B7B7B7; margin: 0;">Safety layer that evaluates AI responses before delivery to patients</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Routing thresholds with info
    st.markdown("### Confidence Thresholds")
    show_info_expander("confidence_threshold", "Info: How confidence routing works")
    thresholds = get_routing_thresholds()
    
    if thresholds:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #2d2d2d; padding: 20px; border-radius: 4px;">
                <h4 style="color: #FFFFFF;">Confidence Levels</h4>
            </div>
            """, unsafe_allow_html=True)
            
            thresh = thresholds.get("thresholds", {})
            for level, value in thresh.items():
                level_display = level.replace("_", " ").title()
                color = "#73BF69" if "high" in level else "#F79420" if "medium" in level else "#E24D42"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #404040;">
                    <span style="color: #B7B7B7;">{level_display}</span>
                    <span style="color: {color}; font-weight: bold;">{value:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #2d2d2d; padding: 20px; border-radius: 4px;">
                <h4 style="color: #FFFFFF;">Risk Level Penalties</h4>
            </div>
            """, unsafe_allow_html=True)
            
            penalties = thresholds.get("topic_penalties", {})
            for level, penalty in penalties.items():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #404040;">
                    <span style="color: #B7B7B7;">{level.title()} Risk</span>
                    <span style="color: #F79420;">-{penalty:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sensitive Topics
    st.markdown("### Sensitive Topic Categories")
    topics = get_sensitive_topics()
    
    if topics:
        cols = st.columns(len(topics))
        risk_colors = {"low": "#73BF69", "medium": "#F79420", "high": "#E24D42", "critical": "#8B0000"}
        
        for i, (topic, config) in enumerate(topics.items()):
            risk_level = config.get("risk_level", "unknown")
            color = risk_colors.get(risk_level, "#808080")
            
            with cols[i]:
                st.markdown(f"""
                <div style="background: #2d2d2d; border-left: 3px solid {color}; padding: 15px; text-align: center;">
                    <div style="font-size: 14px; color: #FFFFFF; font-weight: bold; text-transform: uppercase;">
                        {topic.replace('_', ' ')}
                    </div>
                    <div style="font-size: 12px; color: {color}; margin-top: 8px;">
                        {risk_level.upper()} RISK
                    </div>
                    <div style="font-size: 11px; color: #B7B7B7; margin-top: 4px;">
                        {'Validation Required' if config.get('requires_validation') else 'Standard Processing'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Routing Decision Guide
    st.markdown("### Routing Decision Guide")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px;">
        <div style="background: #2d2d2d; border-left: 4px solid #73BF69; padding: 15px;">
            <div style="color: #73BF69; font-weight: bold;">AI Response</div>
            <div style="color: #B7B7B7; font-size: 12px; margin-top: 8px;">
                High confidence, low risk. Safe to deliver AI response directly.
            </div>
        </div>
        <div style="background: #2d2d2d; border-left: 4px solid #3274D9; padding: 15px;">
            <div style="color: #3274D9; font-weight: bold;">Hold for Review</div>
            <div style="color: #B7B7B7; font-size: 12px; margin-top: 8px;">
                Medium confidence with validation issues. Queue for async human review.
            </div>
        </div>
        <div style="background: #2d2d2d; border-left: 4px solid #F79420; padding: 15px;">
            <div style="color: #F79420; font-weight: bold;">Safe Fallback</div>
            <div style="color: #B7B7B7; font-size: 12px; margin-top: 8px;">
                Low confidence or high risk. Use pre-approved safe response.
            </div>
        </div>
        <div style="background: #2d2d2d; border-left: 4px solid #E24D42; padding: 15px;">
            <div style="color: #E24D42; font-weight: bold;">Human Escalation</div>
            <div style="color: #B7B7B7; font-size: 12px; margin-top: 8px;">
                Critical risk or very low confidence. Route to human agent immediately.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
            show_known_good = st.checkbox(
                "Show only known-good versions", 
                value=False,
                help="Known-good versions have been verified to perform well and are preferred for rollbacks"
            )
        with col2:
            if st.button("Create Snapshot", help="Save current configuration as a new version that can be restored later"):
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
    
    # Automated Triggers Status
    st.markdown("### Automated Rollback Triggers")
    trigger_status = get_trigger_status()
    
    if trigger_status:
        in_cooldown = trigger_status.get("in_cooldown", False)
        cooldown_color = "#F79420" if in_cooldown else "#73BF69"
        cooldown_text = "IN COOLDOWN" if in_cooldown else "ARMED"
        
        st.markdown(f"""
        <div style="background: #2d2d2d; border-left: 4px solid {cooldown_color}; padding: 15px; margin-bottom: 20px;">
            <span style="color: {cooldown_color}; font-weight: bold;">System Status: {cooldown_text}</span>
            <span style="color: #B7B7B7; margin-left: 20px;">
                Last Rollback: {trigger_status.get('last_rollback', 'Never')[:19] if trigger_status.get('last_rollback') else 'Never'}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        triggers = trigger_status.get("triggers", {})
        current_state = trigger_status.get("current_state", {})
        
        st.markdown("**Active Triggers:**")
        cols = st.columns(len(triggers))
        
        for i, (trigger_name, trigger_config) in enumerate(triggers.items()):
            enabled = trigger_config.get("enabled", False)
            color = "#73BF69" if enabled else "#808080"
            
            with cols[i]:
                st.markdown(f"""
                <div style="background: #2d2d2d; padding: 15px; border-radius: 4px; text-align: center;">
                    <div style="color: {color}; font-weight: bold; font-size: 12px; text-transform: uppercase;">
                        {trigger_name.replace('_', ' ')}
                    </div>
                    <div style="color: {'#73BF69' if enabled else '#808080'}; margin-top: 8px;">
                        {'ENABLED' if enabled else 'DISABLED'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("**Current Alert State:**")
        col1, col2 = st.columns(2)
        with col1:
            emergency_count = current_state.get("emergency_alerts_1h", 0)
            st.metric("Emergency Alerts (1h)", emergency_count)
        with col2:
            critical_count = current_state.get("critical_alerts_1h", 0)
            st.metric("Critical Alerts (1h)", critical_count)
    
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
