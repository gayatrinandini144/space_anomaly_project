"""
dashboard/app.py — Premium Mission Control Dashboard
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

st.set_page_config(
    page_title="Space Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── AGGRESSIVE CSS OVERRIDE ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Force dark background everywhere */
html, body, [class*="css"], .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div,
.main, .block-container {
    background-color: #060b18 !important;
    color: #cbd5e1 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1428 0%, #060b18 100%) !important;
    border-right: 1px solid #1e3058 !important;
}

/* Remove default padding */
.block-container {
    padding: 1rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0c1428, #111e36) !important;
    border: 1px solid #1e3058 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: #0c1428 !important;
    border-bottom: 1px solid #1e3058 !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
[aria-selected="true"] {
    background: #1e3058 !important;
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e3058 !important;
}

/* Sliders */
[data-testid="stSlider"] .st-emotion-cache-1dp5vir,
.stSlider [data-baseweb="slider"] {
    background: #1e3058 !important;
}

/* Selectbox / multiselect */
[data-testid="stMultiSelect"] div,
[data-baseweb="select"] div {
    background: #0c1428 !important;
    border-color: #1e3058 !important;
    color: #cbd5e1 !important;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    color: #cbd5e1 !important;
    font-size: 13px !important;
}

/* Divider */
hr { border-color: #1e3058 !important; }

/* Text */
p, li, span { color: #cbd5e1 !important; }
h1 { color: #f1f5f9 !important; font-size: 2rem !important; font-weight: 800 !important; }
h2 { color: #e2e8f0 !important; font-size: 1.3rem !important; font-weight: 700 !important; }
h3 { color: #cbd5e1 !important; font-size: 1.1rem !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
}

/* Alert box custom */
.alert-critical { background:#1a0505; border-left:4px solid #ef4444;
                  border-radius:8px; padding:12px 16px; margin:6px 0; }
.alert-warning  { background:#1a1005; border-left:4px solid #f59e0b;
                  border-radius:8px; padding:12px 16px; margin:6px 0; }
.alert-ok       { background:#0a1a0a; border-left:4px solid #10b981;
                  border-radius:8px; padding:12px 16px; margin:6px 0; }

/* KPI card */
.kpi-card {
    background: linear-gradient(135deg, #0c1428, #111e36);
    border: 1px solid #1e3058;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin: 4px;
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #f1f5f9; }
.kpi-label { font-size: 11px; color: #64748b; font-weight: 600;
             letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e3058 !important; border-radius: 8px; }

/* Caption */
.stCaption p { color: #64748b !important; font-size: 13px !important; }

/* Info / warning / success */
[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

BG   = "#060b18"
SURF = "#0c1428"
BORD = "#1e3058"
SURF2= "#111e36"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_telemetry():
    p = "data/spacecraft_telemetry.csv"
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["timestamp"])
    from utils.data_generator import (generate_spacecraft_telemetry,
                                       add_rul_labels, add_subsystem_health_scores)
    from utils.feature_engineering import handle_missing_noisy_data
    df = generate_spacecraft_telemetry(2000)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    return add_subsystem_health_scores(df)

@st.cache_data
def load_decisions():
    p = "reports/decision_log.json"
    if os.path.exists(p):
        with open(p) as f: return pd.DataFrame(json.load(f))
    return pd.DataFrame()

@st.cache_data
def load_model_results():
    p = "reports/ml_results.pkl"
    if os.path.exists(p):
        import joblib; return joblib.load(p)
    return []

def chart_defaults():
    return dict(paper_bgcolor=BG, plot_bgcolor=SURF,
                font=dict(color="#cbd5e1", size=12),
                margin=dict(t=40,b=20,l=10,r=10))

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0 10px;'>
          <div style='font-size:2.5rem'>🛰️</div>
          <div style='font-size:1.1rem; font-weight:800; color:#3b82f6;
                      letter-spacing:0.1em;'>MISSION CONTROL</div>
          <div style='font-size:10px; color:#64748b; margin-top:4px;'>
               AI ANOMALY DETECTION</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1e3058;margin:8px 0 16px'>",
                    unsafe_allow_html=True)

        page = st.radio("", [
            "🏠  Home Dashboard",
            "📈  Live Telemetry",
            "🔮  Anomaly Predictor",
            "📊  Model Comparison",
            "🤖  Decision Log",
            "🎮  Mission Simulator",
            "🛸  NASA Dataset",
        ], label_visibility="collapsed")

        st.markdown("<hr style='border-color:#1e3058;margin:16px 0'>",
                    unsafe_allow_html=True)
        window = st.slider("Display window", 50, min(3000,len(df)), 300)

        # Live stats
        anom = int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
        risk = df["mission_risk_score"].mean() if "mission_risk_score" in df.columns else 0
        rlabel = "🔴 HIGH" if risk>0.6 else "🟡 MED" if risk>0.35 else "🟢 LOW"
        st.markdown(f"""
        <div style='display:flex;flex-direction:column;gap:8px;margin-top:8px'>
          <div class='kpi-card'><div class='kpi-value' style='color:#ef4444'>{anom}</div>
               <div class='kpi-label'>Anomalies</div></div>
          <div class='kpi-card'><div class='kpi-value' style='font-size:1.3rem'>{rlabel}</div>
               <div class='kpi-label'>Mission Risk</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        st.download_button("⬇️ Export CSV", df.to_csv(index=False).encode(),
                           "telemetry.csv","text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    return page, window

# ── GAUGE ─────────────────────────────────────────────────────────────────────
def gauge(label, val, unit, lo, hi, warn, crit, reverse=False):
    bad  = [crit,hi]  if reverse else [lo,crit]
    mid  = [warn,crit] if reverse else [crit,warn]
    good = [lo,warn]  if reverse else [warn,hi]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={"suffix":unit,"font":{"size":20,"color":"#f1f5f9"}},
        title={"text":label,"font":{"size":11,"color":"#64748b"}},
        gauge={"axis":{"range":[lo,hi],"tickcolor":"#1e3058",
                       "tickfont":{"color":"#64748b","size":9}},
               "bar":{"color":"#3b82f6","thickness":0.3},
               "bgcolor":SURF2,"borderwidth":1,"bordercolor":BORD,
               "steps":[{"range":bad,"color":"rgba(239,68,68,0.25)"},
                        {"range":mid,"color":"rgba(245,158,11,0.2)"},
                        {"range":good,"color":"rgba(16,185,129,0.15)"}]}
    ))
    fig.update_layout(height=220,**chart_defaults())
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home(df, dec_df, window):
    st.markdown("""
    <div style='padding:20px 0 8px'>
      <div style='font-size:2.2rem;font-weight:900;color:#f1f5f9;
                  letter-spacing:-0.02em'>
        🛰️ Deep Space Mission Control
      </div>
      <div style='color:#64748b;font-size:14px;margin-top:4px'>
        AI-Based Anomaly Detection & Autonomous Decision Support System
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    latest = df.iloc[-1].to_dict()
    anom   = int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
    risk   = float(df["mission_risk_score"].iloc[-1]) if "mission_risk_score" in df.columns else 0
    batt   = float(latest.get("battery_charge", 75))
    temp   = float(latest.get("internal_temp",  22))
    fuel   = float(latest.get("fuel_pressure", 150))
    sig    = float(latest.get("signal_strength",-80))
    rlabel = "🔴 HIGH" if risk>0.6 else "🟡 MEDIUM" if risk>0.35 else "🟢 LOW"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("⚠️ Anomalies Detected", anom)
    c2.metric("🎯 Current Risk",   rlabel)
    c3.metric("⚡ Battery",        f"{batt:.1f}%",  f"{batt-75:.1f}%")
    c4.metric("🌡️ Internal Temp",  f"{temp:.1f}°C", f"{temp-22:.1f}")
    c5.metric("🔥 Fuel Pressure",  f"{fuel:.1f}",   f"{fuel-150:.1f}")
    st.divider()

    # Gauges
    st.markdown("### 📊 Subsystem Status")
    g1,g2,g3,g4 = st.columns(4)
    g1.plotly_chart(gauge("⚡ Battery %",   batt,   "%",   0,100,  40,  15),  use_container_width=True)
    g2.plotly_chart(gauge("🌡️ Temp (°C)",   temp,   "°C", -5, 80,  45,  60, reverse=True), use_container_width=True)
    g3.plotly_chart(gauge("🔥 Fuel (PSI)",  fuel,   " ",  50,200, 110,  90),  use_container_width=True)
    g4.plotly_chart(gauge("📡 Signal",abs(sig),"dBm",40,120, 70,  60, reverse=True), use_container_width=True)

    # Risk timeline + alerts side by side
    col_left, col_right = st.columns([3,2])
    with col_left:
        st.markdown("### 📉 Mission Risk Timeline")
        if "mission_risk_score" in df.columns:
            recent = df.tail(window)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["mission_risk_score"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="#ef4444",width=2), name="Risk"))
            fig.add_hrect(y0=0.75,y1=1.0,fillcolor="rgba(239,68,68,0.07)",line_width=0)
            fig.add_hrect(y0=0.50,y1=0.75,fillcolor="rgba(245,158,11,0.05)",line_width=0)
            fig.add_hline(y=0.5, line_dash="dot", line_color="#f59e0b",
                          annotation_text="⚠ WARNING", annotation_font_color="#f59e0b")
            fig.add_hline(y=0.75,line_dash="dot", line_color="#ef4444",
                          annotation_text="🔴 CRITICAL",annotation_font_color="#ef4444")
            fig.update_yaxes(range=[0,1],gridcolor=BORD)
            fig.update_xaxes(gridcolor=BORD)
            fig.update_layout(height=280, showlegend=False, **chart_defaults())
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🚨 Live Alerts")
        if not dec_df.empty:
            crit = dec_df[dec_df["severity"].isin(["CRITICAL","WARNING"])].tail(6)
            if crit.empty:
                st.markdown('<div class="alert-ok"><strong style="color:#10b981">✅ All systems nominal</strong></div>',
                            unsafe_allow_html=True)
            for _,r in crit.iloc[::-1].iterrows():
                is_crit = r["severity"]=="CRITICAL"
                cls = "alert-critical" if is_crit else "alert-warning"
                ico = "🔴" if is_crit else "🟡"
                col = "#ef4444" if is_crit else "#f59e0b"
                st.markdown(f"""<div class="{cls}">
                  <strong style="color:{col}">{ico} {r['severity']} — {r['subsystem']}</strong><br>
                  <small style="color:#94a3b8">{r.get('reasoning','')[:80]}…</small><br>
                  <small style="color:#64748b">Action: <b>{r.get('recommended_action','')}</b></small>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Run `python train.py --phase 3` for alerts")

    # Anomaly type pie
    if "anomaly_type" in df.columns:
        st.divider()
        st.markdown("### 🔬 Anomaly Distribution")
        c1,c2 = st.columns(2)
        with c1:
            vc = df[df["anomaly_label"]==1]["anomaly_type"].value_counts()
            fig = px.pie(names=vc.index, values=vc.values,
                         color_discrete_map={"power_spike":"#f59e0b",
                             "thermal_runaway":"#ef4444",
                             "fuel_leak":"#3b82f6","comm_dropout":"#a78bfa"},
                         hole=0.45)
            fig.update_layout(title="Anomaly Types", height=300, **chart_defaults())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            health = {
                "Power":       df.get("power_health",pd.Series([0.8])).mean(),
                "Thermal":     df.get("thermal_health",pd.Series([0.85])).mean(),
                "Propulsion":  df.get("propulsion_health",pd.Series([0.9])).mean(),
                "Comm":        df.get("comm_health",pd.Series([0.78])).mean(),
            }
            fig = go.Figure(go.Bar(
                x=list(health.keys()), y=list(health.values()),
                marker=dict(color=["#3b82f6","#10b981","#f59e0b","#a78bfa"],
                            line=dict(width=0)),
                text=[f"{v:.2f}" for v in health.values()],
                textposition="outside", textfont=dict(color="#f1f5f9")
            ))
            fig.update_yaxes(range=[0,1.15], gridcolor=BORD)
            fig.update_xaxes(gridcolor=BORD)
            fig.update_layout(title="Subsystem Health Scores",height=300,
                              showlegend=False, **chart_defaults())
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════
def page_telemetry(df, window):
    st.markdown("# 📈 Live Telemetry Monitor")
    st.caption("Real-time sensor data with anomaly markers across all subsystems")
    st.divider()

    recent    = df.tail(window)
    anomalies = recent[recent["anomaly_label"]==1] if "anomaly_label" in recent.columns else pd.DataFrame()

    sensors = st.multiselect("Select sensors to display:", [
        "solar_panel_voltage","battery_charge","power_consumption",
        "internal_temp","external_temp","thermal_control_power",
        "fuel_pressure","thruster_temp","delta_v",
        "signal_strength","data_rate","bit_error_rate"
    ], default=["battery_charge","internal_temp","fuel_pressure","bit_error_rate"])

    if sensors:
        colors = ["#3b82f6","#10b981","#f59e0b","#a78bfa","#ef4444","#06b6d4","#f97316","#84cc16"]
        fig = make_subplots(rows=len(sensors),cols=1,shared_xaxes=True,vertical_spacing=0.02)
        for i,s in enumerate(sensors):
            r = i+1
            fig.add_trace(go.Scatter(
                x=recent["timestamp"],y=recent[s],name=s,
                line=dict(color=colors[i%len(colors)],width=1.5),
                hovertemplate=f"<b>{s}</b>: %{{y:.3f}}<extra></extra>"
            ),row=r,col=1)
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies["timestamp"],y=anomalies[s],
                    mode="markers",name="Anomaly" if i==0 else None,
                    marker=dict(color="#ef4444",size=8,symbol="x",
                                line=dict(width=2,color="#ef4444")),
                    showlegend=(i==0),
                    hovertemplate="<b>ANOMALY</b><extra></extra>"
                ),row=r,col=1)
            fig.update_yaxes(title_text=s,row=r,col=1,
                             gridcolor=BORD,color="#64748b",title_font_size=10)
        fig.update_layout(height=130*len(sensors)+60,showlegend=True,
                          legend=dict(bgcolor=SURF,bordercolor=BORD,x=1.01),
                          **chart_defaults())
        fig.update_xaxes(gridcolor=BORD,color="#64748b")
        st.plotly_chart(fig, use_container_width=True)

    if "anomaly_type" in df.columns:
        st.divider()
        st.markdown("### 🔎 Anomaly Type Timeline")
        anom_df = df[df["anomaly_label"]==1].copy()
        if not anom_df.empty:
            vc = anom_df["anomaly_type"].value_counts()
            fig2 = px.bar(x=vc.index,y=vc.values,
                          color=vc.index,
                          color_discrete_map={"power_spike":"#f59e0b",
                              "thermal_runaway":"#ef4444",
                              "fuel_leak":"#3b82f6","comm_dropout":"#a78bfa"},
                          text=vc.values)
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=300,showlegend=False,**chart_defaults())
            st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
def page_predictor():
    st.markdown("# 🔮 Anomaly Predictor")
    st.caption("Adjust sensor values and get an instant AI-based anomaly prediction")
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚡ Power Subsystem")
        voltage = st.slider("Solar Panel Voltage (V)", 20.0,50.0,28.5,0.1)
        battery = st.slider("Battery Charge (%)",       0.0,100.0,75.0,1.0)
        power   = st.slider("Power Consumption (W)",  300.0,700.0,450.0,5.0)
        st.markdown("#### 🌡️ Thermal Subsystem")
        int_t   = st.slider("Internal Temp (°C)",    -10.0,100.0,22.0,0.5)
        ext_t   = st.slider("External Temp (°C)",   -100.0, 50.0,-50.0,1.0)
        th_pow  = st.slider("Thermal Control (W)",    50.0,200.0,80.0,1.0)

    with col2:
        st.markdown("#### 🔥 Propulsion Subsystem")
        fuel_p  = st.slider("Fuel Pressure (PSI)",   50.0,200.0,150.0,1.0)
        thr_t   = st.slider("Thruster Temp (°C)",   100.0,400.0,200.0,5.0)
        dv      = st.slider("Delta-V (m/s)",           0.0,  2.0,  0.5,0.01)
        st.markdown("#### 📡 Communication")
        sig     = st.slider("Signal Strength (dBm)",-120.0,-40.0,-80.0,1.0)
        drate   = st.slider("Data Rate (Mbps)",       10.0,200.0,100.0,5.0)
        ber_raw = st.slider("Bit Error Rate (×10⁻³)", 0.0,  10.0,  1.0,0.1)
        ber = ber_raw/1000

    st.divider()
    if st.button("🚀 Run Anomaly Prediction", type="primary", use_container_width=True):
        alerts   = []
        risk     = 0.0
        means    = {"solar_panel_voltage":28.5,"battery_charge":60,"power_consumption":450,
                    "internal_temp":22,"external_temp":-50,"thermal_control_power":80,
                    "fuel_pressure":150,"thruster_temp":200,"delta_v":0.5,
                    "signal_strength":-80,"data_rate":100,"bit_error_rate":0.001}
        stds     = {"solar_panel_voltage":0.5,"battery_charge":10,"power_consumption":20,
                    "internal_temp":2,"external_temp":10,"thermal_control_power":5,
                    "fuel_pressure":3,"thruster_temp":10,"delta_v":0.1,
                    "signal_strength":5,"data_rate":10,"bit_error_rate":0.001}
        inputs   = {"solar_panel_voltage":voltage,"battery_charge":battery,
                    "power_consumption":power,"internal_temp":int_t,"external_temp":ext_t,
                    "thermal_control_power":th_pow,"fuel_pressure":fuel_p,
                    "thruster_temp":thr_t,"delta_v":dv,"signal_strength":sig,
                    "data_rate":drate,"bit_error_rate":ber}

        if battery<15:  alerts.append(("CRITICAL","⚡ Battery critically low — enter safe mode!","#ef4444")); risk+=0.9
        elif battery<25:alerts.append(("WARNING", "⚡ Battery low — reduce power mode","#f59e0b")); risk+=0.5
        if int_t>60:    alerts.append(("CRITICAL","🌡️ Thermal runaway — shutdown subsystem!","#ef4444")); risk+=0.9
        elif int_t>45:  alerts.append(("WARNING", "🌡️ High temperature — cooling needed","#f59e0b")); risk+=0.4
        if fuel_p<90:   alerts.append(("CRITICAL","🔥 Fuel pressure critical — possible leak!","#ef4444")); risk+=0.85
        elif fuel_p<110:alerts.append(("WARNING", "🔥 Low fuel pressure — monitor closely","#f59e0b")); risk+=0.4
        if ber>0.05:    alerts.append(("WARNING", "📡 High bit error rate — switch antenna","#f59e0b")); risk+=0.3
        if voltage>40:  alerts.append(("WARNING", "⚡ Voltage spike detected","#f59e0b")); risk+=0.25

        max_z = max(abs(inputs[k]-means[k])/(stds[k]+1e-8) for k in means)
        risk  = min(risk + max_z/25, 1.0)
        label = "🔴 ANOMALY" if risk>0.5 else "🟡 CAUTION" if risk>0.25 else "🟢 NOMINAL"
        col   = "#ef4444"    if risk>0.5 else "#f59e0b"    if risk>0.25 else "#10b981"

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Prediction",  label)
        r2.metric("Risk Score",  f"{risk:.3f}")
        r3.metric("Max Z-Score", f"{max_z:.2f}σ")
        r4.metric("Alerts",      len(alerts))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",value=risk*100,
            number={"suffix":"%","font":{"size":32,"color":col}},
            title={"text":"Mission Risk Score","font":{"size":14,"color":"#94a3b8"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#1e3058"},
                   "bar":{"color":col,"thickness":0.3},
                   "bgcolor":SURF2,"borderwidth":1,"bordercolor":BORD,
                   "steps":[{"range":[0,30],"color":"rgba(16,185,129,0.15)"},
                             {"range":[30,60],"color":"rgba(245,158,11,0.15)"},
                             {"range":[60,100],"color":"rgba(239,68,68,0.15)"}]}
        ))
        fig.update_layout(height=300,**chart_defaults())
        st.plotly_chart(fig,use_container_width=True)

        if alerts:
            st.markdown("### ⚠️ Alert Details")
            for sev,msg,c in alerts:
                cls = "alert-critical" if sev=="CRITICAL" else "alert-warning"
                st.markdown(f'<div class="{cls}"><strong style="color:{c}">{sev}</strong> — {msg}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-ok"><strong style="color:#10b981">✅ All parameters within normal range. Mission nominal.</strong></div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def page_model_comparison(results):
    st.markdown("# 📊 Model Comparison Report")
    st.caption("Performance metrics across all trained anomaly detection models")
    st.divider()

    if not results:
        st.warning("⚠️ No model results. Run `python train.py --phase 2` first.")
        return

    df_r = pd.DataFrame(results)
    st.markdown("### 📋 Metrics Table")
    cols = [c for c in ["model","roc_auc","precision","recall",
                         "false_alarm_rate","tp","fp","fn","tn"] if c in df_r.columns]
    fmt  = {c:"{:.4f}" for c in ["roc_auc","precision","recall","false_alarm_rate"] if c in df_r.columns}
    st.dataframe(df_r[cols].style.format(fmt)
                 .highlight_max(subset=[c for c in ["roc_auc","precision","recall"] if c in df_r.columns],
                                color="#0f2e0f")
                 .highlight_min(subset=[c for c in ["false_alarm_rate"] if c in df_r.columns],
                                color="#0f2e0f"),
                 use_container_width=True)

    st.markdown("### 📈 Visual Comparison")
    metrics = [c for c in ["roc_auc","precision","recall","false_alarm_rate"] if c in df_r.columns]
    cols    = st.columns(len(metrics))
    colors  = ["#3b82f6","#10b981","#f59e0b","#ef4444"]
    for i,(m,col) in enumerate(zip(metrics,cols)):
        vals = df_r[m].fillna(0)
        fig = go.Figure(go.Bar(x=df_r["model"],y=vals,
            marker=dict(color=colors[i],line=dict(width=0)),
            text=vals.round(3),textposition="outside",
            textfont=dict(color="#f1f5f9")))
        fig.update_yaxes(range=[0,1.15],gridcolor=BORD)
        fig.update_xaxes(tickangle=-15)
        fig.update_layout(title=m.replace("_"," ").upper(),height=320,
                          showlegend=False,**chart_defaults())
        col.plotly_chart(fig,use_container_width=True)

    st.markdown("### 🎯 Radar Chart")
    radar_m = [c for c in ["roc_auc","precision","recall"] if c in df_r.columns]
    if radar_m:
        fig = go.Figure()
        pal = ["#3b82f6","#10b981","#f59e0b"]
        for i,row in df_r.iterrows():
            vals = [float(row.get(m,0) or 0) for m in radar_m]
            fig.add_trace(go.Scatterpolar(
                r=vals+[vals[0]],theta=radar_m+[radar_m[0]],
                fill="toself",name=row["model"],
                line=dict(color=pal[i%3],width=2),
                fillcolor=pal[i%3].replace("#","rgba(").rstrip()+"0.15)"\
                    if False else f"rgba({int(pal[i%3][1:3],16)},{int(pal[i%3][3:5],16)},{int(pal[i%3][5:],16)},0.15)"
            ))
        fig.update_layout(
            polar=dict(bgcolor=SURF2,
                       radialaxis=dict(range=[0,1],color="#64748b",gridcolor=BORD),
                       angularaxis=dict(color="#94a3b8",gridcolor=BORD)),
            height=420,legend=dict(bgcolor=SURF,bordercolor=BORD),
            **chart_defaults())
        st.plotly_chart(fig,use_container_width=True)

    if os.path.exists("reports/model_comparison.png"):
        st.markdown("### 🖼️ Detailed Evaluation Plots")
        st.image("reports/model_comparison.png",use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DECISIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_decisions(dec_df):
    st.markdown("# 🤖 Autonomous Decision Log")
    st.caption("Real-time rule-based and AI-generated corrective actions")
    st.divider()

    if dec_df.empty:
        st.info("Run `python train.py --phase 3` to generate decisions.")
        return

    c1,c2 = st.columns(2)
    sev_f = c1.multiselect("Severity filter:",
        ["CRITICAL","WARNING","CAUTION","NOMINAL"],["CRITICAL","WARNING"])
    sub_f = c2.multiselect("Subsystem filter:",
        dec_df["subsystem"].unique().tolist(),
        dec_df["subsystem"].unique().tolist())

    filtered = dec_df[dec_df["severity"].isin(sev_f) & dec_df["subsystem"].isin(sub_f)]
    icons = {"CRITICAL":"🔴","WARNING":"🟡","CAUTION":"🟠","NOMINAL":"🟢"}

    for _,r in filtered.tail(12).iloc[::-1].iterrows():
        sev  = r.get("severity","NOMINAL")
        col  = {"CRITICAL":"#ef4444","WARNING":"#f59e0b",
                "CAUTION":"#f97316","NOMINAL":"#10b981"}.get(sev,"#64748b")
        cls  = "alert-critical" if sev=="CRITICAL" else \
               "alert-warning"  if sev=="WARNING"  else "alert-ok"
        st.markdown(f"""<div class="{cls}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong style="color:{col}">{icons.get(sev,'⚪')} {sev} — {r.get('subsystem','')}</strong>
            <small style="color:#64748b">{r.get('timestamp','')}</small>
          </div>
          <div style="font-size:13px;margin-top:4px;color:#cbd5e1">{r.get('reasoning','')}</div>
          <div style="margin-top:4px">
            <span style="background:#1e3058;padding:2px 8px;border-radius:4px;
                         font-size:11px;color:#94a3b8">
              {r.get('recommended_action','')}
            </span>
            <span style="margin-left:8px;font-size:11px;color:#64748b">
              Risk: <b style="color:#f1f5f9">{float(r.get('risk_score',0)):.3f}</b> |
              Conf: <b style="color:#f1f5f9">{float(r.get('confidence',0)):.0%}</b>
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    c1,c2,c3 = st.columns(3)
    with c1:
        vc = dec_df["severity"].value_counts()
        fig=px.pie(names=vc.index,values=vc.values,title="Severity Split",hole=0.4,
                   color=vc.index,color_discrete_map={"CRITICAL":"#ef4444",
                       "WARNING":"#f59e0b","CAUTION":"#f97316","NOMINAL":"#10b981"})
        fig.update_layout(height=280,**chart_defaults())
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        vc2=dec_df["recommended_action"].value_counts()
        fig2=px.bar(x=vc2.values,y=vc2.index,orientation="h",title="Actions",
                    color=vc2.values,color_continuous_scale="Blues")
        fig2.update_layout(height=280,coloraxis_showscale=False,**chart_defaults())
        st.plotly_chart(fig2,use_container_width=True)
    with c3:
        vc3=dec_df["subsystem"].value_counts()
        fig3=px.pie(names=vc3.index,values=vc3.values,title="By Subsystem",hole=0.4)
        fig3.update_layout(height=280,**chart_defaults())
        st.plotly_chart(fig3,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_simulator():
    st.markdown("# 🎮 Mission Simulator")
    st.caption("Trigger anomalies and test the autonomous response system — save your mission!")
    st.divider()

    if "sim" not in st.session_state:
        st.session_state.sim = {"battery":85.0,"temp":22.0,"fuel":150.0,"signal":-80.0}
    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []
    if "score" not in st.session_state:
        st.session_state.score = 100

    h = st.session_state.sim
    score = max(0, st.session_state.score)

    # Header stats
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("⚡ Battery",  f"{h['battery']:.0f}%")
    c2.metric("🌡️ Temp",     f"{h['temp']:.1f}°C")
    c3.metric("🔥 Fuel",     f"{h['fuel']:.0f} PSI")
    c4.metric("📡 Signal",   f"{h['signal']:.0f} dBm")
    col_s = "#10b981" if score>60 else "#f59e0b" if score>30 else "#ef4444"
    c5.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value' style='color:{col_s}'>{score}</div>
        <div class='kpi-label'>Mission Score</div></div>""",
        unsafe_allow_html=True)

    st.markdown(f"**Mission Health:**")
    st.progress(score, text=f"{score}/100")
    st.divider()

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown("#### ⚡ Trigger Anomaly")
        b1,b2,b3,b4 = st.columns(4)
        if b1.button("💥 Power Spike",  use_container_width=True):
            h["battery"]=max(5, h["battery"]-35); h["temp"]=min(90,h["temp"]+5)
            st.session_state.score-=20
            st.session_state.sim_log.append("💥 Power spike — battery -35%!")
        if b2.button("🔥 Heat Surge",   use_container_width=True):
            h["temp"]=min(90,h["temp"]+30)
            st.session_state.score-=25
            st.session_state.sim_log.append("🔥 Thermal surge — temp +30°C!")
        if b3.button("⛽ Fuel Leak",    use_container_width=True):
            h["fuel"]=max(40,h["fuel"]-60)
            st.session_state.score-=30
            st.session_state.sim_log.append("⛽ Fuel leak — pressure -60 PSI!")
        if b4.button("📡 Comm Loss",    use_container_width=True):
            h["signal"]=min(-50,h["signal"]+25)
            st.session_state.score-=15
            st.session_state.sim_log.append("📡 Comm dropout — signal lost!")

        st.markdown("#### 🛠️ Apply Fix")
        f1,f2,f3,f4 = st.columns(4)
        if f1.button("🔋 Charge",       use_container_width=True):
            h["battery"]=min(100,h["battery"]+25); st.session_state.score=min(100,st.session_state.score+10)
            st.session_state.sim_log.append("✅ Battery charged +25%")
        if f2.button("❄️ Cool",          use_container_width=True):
            h["temp"]=max(15,h["temp"]-20); st.session_state.score=min(100,st.session_state.score+12)
            st.session_state.sim_log.append("✅ Cooling activated -20°C")
        if f3.button("⛽ Pressurise",    use_container_width=True):
            h["fuel"]=min(160,h["fuel"]+30); st.session_state.score=min(100,st.session_state.score+15)
            st.session_state.sim_log.append("✅ Tank pressurised +30 PSI")
        if f4.button("📡 Switch Ant",   use_container_width=True):
            h["signal"]=max(-85,h["signal"]-15); st.session_state.score=min(100,st.session_state.score+8)
            st.session_state.sim_log.append("✅ Backup antenna active")

        if st.button("🔄 Reset Mission", use_container_width=True):
            st.session_state.sim={"battery":85.0,"temp":22.0,"fuel":150.0,"signal":-80.0}
            st.session_state.sim_log=[]; st.session_state.score=100; st.rerun()

        # Score history chart (mini)
        if len(st.session_state.sim_log) > 0:
            st.markdown("#### 📋 Event Log")
            for entry in reversed(st.session_state.sim_log[-8:]):
                color = "#ef4444" if any(x in entry for x in ["spike","surge","leak","dropout"]) else "#10b981"
                st.markdown(f'<div style="font-size:12px;color:{color};padding:2px 0">• {entry}</div>',
                            unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 🤖 AI Response")
        risk=0; actions=[]
        if h["battery"]<15:  risk+=0.4; actions.append(("🔴","Enter Safe Mode","#ef4444"))
        elif h["battery"]<30:risk+=0.2; actions.append(("🟡","Reduce Power","#f59e0b"))
        if h["temp"]>60:     risk+=0.4; actions.append(("🔴","Shutdown Thermal","#ef4444"))
        elif h["temp"]>45:   risk+=0.2; actions.append(("🟡","Activate Cooling","#f59e0b"))
        if h["fuel"]<90:     risk+=0.35;actions.append(("🔴","Shutdown Propulsion","#ef4444"))
        elif h["fuel"]<110:  risk+=0.15;actions.append(("🟡","Monitor Fuel","#f59e0b"))
        if h["signal"]>-60:  risk+=0.15;actions.append(("🟡","Switch Antenna","#f59e0b"))
        risk = min(risk,1.0)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",value=risk*100,
            number={"suffix":"%","font":{"size":28,
                    "color":"#ef4444" if risk>0.6 else "#f59e0b" if risk>0.3 else "#10b981"}},
            title={"text":"Live Risk Score","font":{"size":12,"color":"#94a3b8"}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#ef4444" if risk>0.6 else "#f59e0b" if risk>0.3 else "#10b981"},
                   "bgcolor":SURF2,"bordercolor":BORD,
                   "steps":[{"range":[0,30],"color":"rgba(16,185,129,0.15)"},
                             {"range":[30,60],"color":"rgba(245,158,11,0.15)"},
                             {"range":[60,100],"color":"rgba(239,68,68,0.15)"}]}
        ))
        fig.update_layout(height=260,**chart_defaults())
        st.plotly_chart(fig,use_container_width=True)

        if actions:
            st.markdown("**Recommended Actions:**")
            for ico,act,col in actions:
                st.markdown(f'<div style="background:#0c1428;border:1px solid #1e3058;'
                            f'border-radius:6px;padding:8px 12px;margin:4px 0;font-size:13px">'
                            f'{ico} <span style="color:{col}"><b>{act}</b></span></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-ok"><b style="color:#10b981">✅ All systems nominal</b></div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NASA
# ══════════════════════════════════════════════════════════════════════════════
def page_nasa():
    st.markdown("# 🛸 NASA Dataset Integration")
    st.caption("NASA CMAPSS Turbofan Engine Degradation Dataset")
    st.divider()

    nasa_path = "data/nasa_cmapss_processed.csv"
    if os.path.exists(nasa_path):
        nasa = pd.read_csv(nasa_path)
        st.success(f"✅ NASA CMAPSS loaded: **{len(nasa):,} rows** | **{nasa['unit'].nunique()} engines**")
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows",       f"{len(nasa):,}")
        c2.metric("Engines",    nasa["unit"].nunique())
        c3.metric("Anomaly Rate",f"{nasa['anomaly_label'].mean():.1%}")

        st.markdown("### 📈 RUL Per Engine")
        units = nasa["unit"].unique()[:6]
        fig=go.Figure()
        colors=["#3b82f6","#10b981","#f59e0b","#a78bfa","#ef4444","#06b6d4"]
        for i,u in enumerate(units):
            sub=nasa[nasa["unit"]==u]
            fig.add_trace(go.Scatter(x=sub["cycle"],y=sub["rul"],
                name=f"Engine {u}",line=dict(color=colors[i],width=2)))
        fig.update_layout(title="Remaining Useful Life per Engine",height=340,
                          xaxis_title="Cycle",yaxis_title="RUL",**chart_defaults())
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("NASA CMAPSS dataset not found.")
        st.markdown("""
        ### 📥 How to Download NASA Dataset

        1. Go to 👉 **https://www.kaggle.com/datasets/behrad3d/nasa-cmaps**
        2. Click **Download** (free Kaggle account required)
        3. Extract the ZIP — find `train_FD001.txt`
        4. Inside your project, create folder: `data/nasa_cmapss/`
        5. Place `train_FD001.txt` inside that folder
        6. Run: `python train.py --phase 1`
        7. Come back to this page — data loads automatically ✅
        """)
        st.info("All other dashboard pages work perfectly with the simulated data.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df      = load_telemetry()
    dec_df  = load_decisions()
    results = load_model_results()
    page, window = sidebar(df)

    page = page.strip()
    if   "Home"       in page: page_home(df, dec_df, window)
    elif "Telemetry"  in page: page_telemetry(df, window)
    elif "Predictor"  in page: page_predictor()
    elif "Comparison" in page: page_model_comparison(results)
    elif "Decision"   in page: page_decisions(dec_df)
    elif "Simulator"  in page: page_simulator()
    elif "NASA"       in page: page_nasa()

    st.markdown("---")
    st.caption("🛰️ AI & ML in Space Exploration — Enhanced Mission Control v2.0")

if __name__=="__main__":
    main()
