"""
dashboard/app.py
Phase 4 - Days 18-20: Mission Control Dashboard
Run with: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Space Anomaly Detection – Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; color: #e0e6ff; }
    .stApp { background-color: #0a0e1a; }
    .metric-card {
        background: linear-gradient(135deg, #1a2035, #0d1526);
        border: 1px solid #2a3a6e;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 6px 0;
    }
    .alert-critical { border-left: 5px solid #ff4444; background: #2a1010; padding: 10px; border-radius: 5px; }
    .alert-warning  { border-left: 5px solid #ffaa00; background: #2a2010; padding: 10px; border-radius: 5px; }
    .alert-nominal  { border-left: 5px solid #00cc66; background: #0a2010; padding: 10px; border-radius: 5px; }
    .header-title { font-size: 2.2rem; font-weight: 800; color: #4db8ff; letter-spacing: 2px; }
    .sub-title    { font-size: 1rem; color: #7a9cc5; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_telemetry():
    path = "data/spacecraft_telemetry.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    # Generate on-the-fly if not trained yet
    from utils.data_generator import generate_spacecraft_telemetry, add_rul_labels
    from utils.feature_engineering import handle_missing_noisy_data, add_subsystem_health_scores
    df = generate_spacecraft_telemetry(1000)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    df = add_subsystem_health_scores(df)
    return df


@st.cache_data
def load_decisions():
    path = "reports/decision_log.json"
    if os.path.exists(path):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


# ── Header ────────────────────────────────────────────────────────────────────
def render_header(df):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown('<div class="header-title">🛰️ DEEP SPACE MISSION CONTROL</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">AI-Based Real-Time Anomaly Detection & Autonomous Decision Support</div>', unsafe_allow_html=True)
    with col2:
        anomaly_count = df["anomaly_label"].sum() if "anomaly_label" in df.columns else 0
        st.metric("🔴 Anomalies Detected", int(anomaly_count))
    with col3:
        st.metric("📡 Telemetry Points", len(df))
    st.markdown("---")


# ── Live Sensor Gauges ────────────────────────────────────────────────────────

def render_gauges(latest):
    st.subheader("📊 Live Subsystem Status")
    cols = st.columns(4)

    gauges = [
        ("⚡ Battery", latest.get("battery_charge", 75), "%", 0, 100, 25, 50),
        ("🌡️ Internal Temp", latest.get("internal_temp", 22), "°C", -10, 80, 45, 60),
        ("🔥 Fuel Pressure", latest.get("fuel_pressure", 150), "PSI", 50, 200, 90, 130),
        ("📶 Signal", latest.get("signal_strength", -80), "dBm", -120, -40, -60, -70),
    ]

    for i, (label, val, unit, lo, hi, warn, crit) in enumerate(gauges):
        with cols[i]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix": unit, "font": {"size": 20, "color": "#e0e6ff"}},
                title={"text": label, "font": {"size": 13, "color": "#7a9cc5"}},
                gauge={
                    "axis": {"range": [lo, hi], "tickcolor": "#7a9cc5"},
                    "bar": {"color": "#4db8ff"},
                    "bgcolor": "#1a2035",
                    "bordercolor": "#2a3a6e",
                    "steps": [
                        {"range": [lo, crit], "color": "#ff4444"},
                        {"range": [crit, warn], "color": "#ffaa00"},
                        {"range": [warn, hi], "color": "#1a3a2a"},
                    ],
                }
            ))
            fig.update_layout(
                height=220, paper_bgcolor="#0a0e1a",
                font={"color": "#e0e6ff"},
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Time Series Panel ────────────────────────────────────────────────────────

def render_timeseries(df, window=200):
    st.subheader("📈 Telemetry Time Series")
    recent = df.tail(window).copy()
    anomalies = recent[recent["anomaly_label"] == 1]

    sensors = st.multiselect(
        "Select sensors to display:",
        ["solar_panel_voltage", "battery_charge", "power_consumption",
         "internal_temp", "fuel_pressure", "signal_strength", "bit_error_rate"],
        default=["battery_charge", "internal_temp", "fuel_pressure"]
    )

    if not sensors:
        return

    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True,
                        vertical_spacing=0.04)

    colors = ["#4db8ff", "#00cc99", "#ff9900", "#cc44ff", "#ff6666", "#ffcc00"]
    for i, sensor in enumerate(sensors):
        r = i + 1
        fig.add_trace(
            go.Scatter(x=recent["timestamp"], y=recent[sensor],
                       name=sensor, line=dict(color=colors[i % len(colors)], width=1.5)),
            row=r, col=1
        )
        if len(anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["timestamp"], y=anomalies[sensor],
                    mode="markers", name="Anomaly",
                    marker=dict(color="red", size=8, symbol="x"),
                    showlegend=(i == 0)
                ),
                row=r, col=1
            )
        fig.update_yaxes(title_text=sensor, row=r, col=1,
                         gridcolor="#1a2035", color="#7a9cc5")

    fig.update_layout(
        height=120 * len(sensors) + 80,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1526",
        font=dict(color="#e0e6ff"),
        legend=dict(bgcolor="#1a2035", bordercolor="#2a3a6e"),
        margin=dict(t=20, b=20)
    )
    fig.update_xaxes(gridcolor="#1a2035", color="#7a9cc5")
    st.plotly_chart(fig, use_container_width=True)


# ── Risk Score Panel ─────────────────────────────────────────────────────────

def render_risk(df, window=200):
    st.subheader("⚠️ Mission Risk Score")
    recent = df.tail(window)
    if "mission_risk_score" not in recent.columns:
        st.info("Run train.py first to generate risk scores.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["timestamp"], y=recent["mission_risk_score"],
        fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
        line=dict(color="#ff4444", width=2), name="Mission Risk"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ffaa00",
                  annotation_text="WARNING", annotation_position="right")
    fig.add_hline(y=0.75, line_dash="dash", line_color="#ff4444",
                  annotation_text="CRITICAL", annotation_position="right")

    fig.update_layout(
        height=260, paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1526",
        font=dict(color="#e0e6ff"), yaxis=dict(range=[0, 1]),
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Decision Log Panel ───────────────────────────────────────────────────────

def render_decisions(decisions_df):
    st.subheader("🤖 Autonomous Decision Log")
    if decisions_df.empty:
        st.info("No decisions yet. Run train.py --phase 3 to generate decisions.")
        return

    severity_colors = {
        "CRITICAL": "🔴", "WARNING": "🟡", "CAUTION": "🟠", "NOMINAL": "🟢"
    }

    for _, row in decisions_df.tail(10).iloc[::-1].iterrows():
        icon = severity_colors.get(row.get("severity", "NOMINAL"), "⚪")
        css_class = {
            "CRITICAL": "alert-critical", "WARNING": "alert-warning"
        }.get(row.get("severity", ""), "alert-nominal")

        st.markdown(f"""
        <div class="{css_class}">
          <strong>{icon} [{row.get('severity','?')}] {row.get('subsystem','?').upper()}</strong>
          &nbsp;·&nbsp; <em>{row.get('timestamp','')}</em><br>
          {row.get('reasoning', '')} <br>
          <small>Action: <strong>{row.get('recommended_action','?')}</strong>
          | Risk: {row.get('risk_score',0):.2f}
          | Confidence: {row.get('confidence',0):.0%}</small>
        </div><br>
        """, unsafe_allow_html=True)

    # Summary bar chart
    action_counts = decisions_df["recommended_action"].value_counts()
    fig = px.bar(x=action_counts.index, y=action_counts.values,
                 color=action_counts.values,
                 color_continuous_scale="RdYlGn_r",
                 labels={"x": "Action", "y": "Count"})
    fig.update_layout(
        title="Action Distribution",
        height=280, paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1526",
        font=dict(color="#e0e6ff"), coloraxis_showscale=False,
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Anomaly Distribution ──────────────────────────────────────────────────────

def render_anomaly_distribution(df):
    st.subheader("🔬 Anomaly Analysis")
    c1, c2 = st.columns(2)

    with c1:
        labels = ["Normal", "Anomaly"]
        vals = [
            (df["anomaly_label"] == 0).sum(),
            (df["anomaly_label"] == 1).sum()
        ]
        fig = px.pie(names=labels, values=vals,
                     color_discrete_sequence=["#00cc66", "#ff4444"])
        fig.update_layout(
            title="Class Distribution",
            height=280, paper_bgcolor="#0a0e1a",
            font=dict(color="#e0e6ff"), margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        subsystem_scores = {
            "Power":         df.get("power_health", pd.Series([0.8])).mean(),
            "Thermal":       df.get("thermal_health", pd.Series([0.85])).mean(),
            "Propulsion":    df.get("propulsion_health", pd.Series([0.9])).mean(),
            "Communication": df.get("comm_health", pd.Series([0.78])).mean(),
        }
        fig = go.Figure(go.Bar(
            x=list(subsystem_scores.keys()),
            y=list(subsystem_scores.values()),
            marker_color=["#4db8ff", "#00cc99", "#ff9900", "#cc44ff"]
        ))
        fig.update_layout(
            title="Average Subsystem Health",
            yaxis=dict(range=[0, 1]),
            height=280, paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1526",
            font=dict(color="#e0e6ff"), margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🛰️ Mission Settings")
        st.markdown("---")
        window = st.slider("Time Window (samples)", 50, len(df), 200)
        st.markdown("---")
        st.markdown("### 📋 Quick Stats")
        if "anomaly_label" in df.columns:
            rate = df["anomaly_label"].mean()
            st.metric("Anomaly Rate", f"{rate:.1%}")
        if "mission_risk_score" in df.columns:
            risk = df["mission_risk_score"].mean()
            st.metric("Avg Mission Risk", f"{risk:.3f}")
        st.markdown("---")
        st.markdown("### 📁 Data")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        st.download_button(
            "⬇️ Download Telemetry CSV",
            data=df.to_csv(index=False).encode(),
            file_name="telemetry_export.csv",
            mime="text/csv"
        )
    return window


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_telemetry()
    decisions_df = load_decisions()

    window = render_sidebar(df)
    render_header(df)

    latest = df.iloc[-1].to_dict()
    render_gauges(latest)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Telemetry", "⚠️ Risk Scores", "🤖 Decisions", "🔬 Analysis"]
    )

    with tab1:
        render_timeseries(df, window)
    with tab2:
        render_risk(df, window)
    with tab3:
        render_decisions(decisions_df)
    with tab4:
        render_anomaly_distribution(df)

    st.markdown("---")
    st.markdown(
        "<center><small style='color:#2a3a6e;'>AI & ML in Space Exploration — "
        "Anomaly Detection System | Built with Streamlit + Plotly</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
